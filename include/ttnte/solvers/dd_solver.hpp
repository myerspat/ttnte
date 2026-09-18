#pragma once

#include "ttnte/linalg/format_type.hpp"
#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/linalg/tt_config.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/boundary_communicator.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/parallel/request.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/solvers/solver.hpp"
#include "ttnte/solvers/solver_configs.hpp"
#include "ttnte/task/task_graph.hpp"
#include "ttnte/task/task_scheduler.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/label.hpp"

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace ttnte::solvers {

/// @brief The domain decomposition solver class.
template<typename BlockType>
class DDSolver : public Solver {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<DDSolver>;
  using Ptr = std::shared_ptr<DDSolver>;
  using Mesh = mesh::Mesh<BlockType>;
  using Callback = std::function<void(const DDSolver&)>;

private:
  // =================================================================
  // Private data
  /// Label of the DD solver.
  Label label_;
  /// Shared pointer to the mesh.
  Mesh::Ptr mesh_;
  /// All the linear systems on this MPI rank.
  std::vector<linalg::LinearSystem::Ptr> local_systems_;
  /// Map from mesh-block GID to index in local_systems_.
  std::unordered_map<int64_t, size_t> gid_to_local_idx_;
  /// The solver strategy (also owns DDSolverConfig).
  DDStrategy::Ptr strategy_;
  /// DAG for the solver.
  task::TaskGraph dag_;
  /// GPU device.
  const torch::Device device_;

  /// The world communicator.
  parallel::Communicator world_comm_;
  /// The boundary communicators.
  parallel::BoundaryCommunicator boundary_comms_;
  /// Scheduler for executing the solver DAG. Owns its own GPU stream pool
  /// (see TaskScheduler) -- DDSolver has no direct use for the streams
  /// itself, only for sizing/passing through to the scheduler.
  task::TaskScheduler scheduler_;

  // State variables
  bool is_initialized_ = false;
  bool is_finalized_ = false;
  bool is_converged_ = false;
  /// Number of Schwarz iterations the most recent step() call ran.
  int last_num_iterations_ = 0;
  /// The Schwarz L2 error at each iteration of the most recent step() call,
  /// in iteration order.
  std::vector<double> last_errors_;
  /// Optional user callback invoked once per Schwarz sweep -- see
  /// set_callback().
  Callback callback_ = nullptr;
  /// Only invoke callback_ every callback_frequency_-th Schwarz sweep (j %
  /// callback_frequency_ == 0) -- see set_callback().
  int callback_frequency_ = 1;

  // =================================================================
  // Private constructors
  DDSolver(Mesh::Ptr mesh, DDStrategy::Ptr strategy,
    std::optional<std::string> label = std::nullopt)
    : mesh_(std::move(mesh)), strategy_(std::move(strategy)),
      device_(parallel::ParallelContext::instance().device()),
      world_comm_(parallel::Communicator::world()),
      boundary_comms_(world_comm_, 2 * mesh_->get_ndim()),
      scheduler_(strategy_->get_config().num_threads),
      label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}

  // =================================================================
  // Private methods
  [[nodiscard]] static std::string error_context(const std::string& func_name)
  {
    return "ttnte::solvers::DDSolver::" + func_name;
  }

public:
  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new instance of the DDSolver.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new DDSolver<BlockType>(std::forward<Args>(args)...));
  }

  /// @brief Block until every worker thread has completed its one-time
  /// initialization (e.g. CUDA device setup). Call this before the first
  /// step() to ensure all threads are ready.
  void wait_for_thread_init() override { scheduler_.wait_for_init(); }

  /// @brief Build one iteration of the DAG using the stored strategy.
  /// @param dag The task graph to populate.
  void build_iteration_dag()
  {
    if (!is_initialized_ || is_finalized_) {
      throw utils::runtime_error(*this, error_context("build_iteration_dag"),
        "The solver has not been initialized yet or has "
        "already been finalized");
    }
    dag_.clear();

    if (strategy_->get_config().use_gpu) {
      strategy_->build_gpu_iteration_dag(
        dag_, local_systems_, gid_to_local_idx_, boundary_comms_);
    } else {
      strategy_->build_cpu_iteration_dag(
        dag_, local_systems_, gid_to_local_idx_, boundary_comms_);
    }
  }

  /// @brief Initialize the DD solver. This method will send systems to GPU
  /// depending on the MemoryPolicy.
  /// @param local_systems The vector of systems local to this MPI rank.
  void init(const Systems& local_systems) override
  {
    // Get the memory policy
    auto memory_policy = strategy_->get_config().memory_policy;

    // Push the linear systems to GPU if needed based on the memory policy
    local_systems_ = local_systems;
    gid_to_local_idx_.clear();
    for (size_t i = 0; i < local_systems_.size(); ++i) {
      gid_to_local_idx_[local_systems_[i]->get_gid()] = i;
      auto& sys = local_systems[i];

      // Move data based on memory policy
      if (memory_policy == MemoryPolicy::RESIDENT) {
        sys->to_(device_);
      } else if (memory_policy == MemoryPolicy::OPERATOR_RESIDENT) {
        sys->transfer_buffer(device_);
      } else if (memory_policy == MemoryPolicy::STATE_RESIDENT) {
        sys->transfer_nonbuffer(device_);
      }
    }

    is_initialized_ = true;
    is_finalized_ = false;
    is_converged_ = false;
    build_iteration_dag();
  }

  /// @brief Run the solver.
  void step() override
  {
    if (!is_initialized_ || is_finalized_) {
      throw utils::runtime_error(*this, error_context("step"),
        "The solver has not been initialized yet or has "
        "already been finalized");
    }

    double error = std::numeric_limits<double>::max();
    parallel::Request ereq;

    const auto& cfg = strategy_->get_config();
    bool verbose = cfg.verbose && world_comm_.rank() == 0;

    // Snapshot the Schwarz break tolerance ONCE for this whole step() call,
    // from min_error_ as it stood at the end of the PREVIOUS step() call
    // (persists across calls, so this still tightens over repeated outer
    // iterations). Deliberately NOT recomputed inside the loop below --
    // doing so would make tol chase the very error it's compared against
    // (tol == tol_forcing * this iteration's own error), so the loop could
    // only ever break once tol bottoms out at the hard floor cfg.tol.
    const double tol =
      std::max(cfg.tol, cfg.tol_forcing * strategy_->get_min_error());

    if (cfg.use_gpu && torch::cuda::is_available()) {
      torch::cuda::synchronize();
    }
    auto start = std::chrono::high_resolution_clock::now();

    last_errors_.clear();
    for (int j = 0; j < cfg.max_iter; j++) {
      // Set before the possible break below so it always reflects the
      // number of iterations actually executed (0..j inclusive), whether
      // this loop exits via convergence or by exhausting max_iter.
      last_num_iterations_ = j + 1;

      // Execute the DAG
      scheduler_.execute(dag_);

      // Compute the total squared L2 norms for the difference between this
      // iteration and last as well as last iteration's partial current at
      // each INTERNAL boundary (NeighborCoupling::sq_diff/sq_prev, computed
      // in LocalSolver::postsolve() from coupling.current_op), and the total
      // TT-state element count across every local patch (for an
      // EnrichmentPolicy's rank-growth backoff, e.g.
      // solvers::AdaptiveRevalidationPolicy -- summed/reduced here rather
      // than inside AMEnSolver since that only ever sees one patch at a
      // time).
      double local_sums[3] = {0.0, 0.0, 0.0};
      for (const auto& sys : local_systems_) {
        for (const auto& coupling : sys->get_couplings()) {
          local_sums[0] += coupling.sq_diff;
          local_sums[1] += coupling.sq_prev;
        }
        local_sums[2] += static_cast<double>(sys->get_state().get_numel());
      }

      // Sum the local norms with all MPI ranks
      double global_sums[3] = {local_sums[0], local_sums[1], local_sums[2]};
      if (world_comm_.size() > 1) {
        ereq = world_comm_.iallreduce(
          local_sums, global_sums, 3, parallel::MPIOp::SUM);
      }

      // Reset the DAG
      dag_.reset();

      // Wait for MPI communication
      ereq.wait();

      // Compute the global error of the DD solver
      error = (global_sums[1] > 0.0)
                ? std::sqrt(global_sums[0] / global_sums[1])
                : 0.0;
      last_errors_.push_back(error);

      // Invoke the user callback (if any, and due this sweep per
      // callback_frequency_) BEFORE tightening the local solver's forcing
      // below -- callback_ observes this via get_eps(), which must still
      // reflect the tolerance that produced this sweep's local_systems_
      // state, not the next sweep's tightened value. The frequency check
      // happens here, before the (possibly Python-wrapping) callback_ is
      // even reached, so a skipped sweep never pays for a GIL acquire.
      if (callback_ && j % callback_frequency_ == 0) {
        callback_(*this);
      }

      // Tighten TT truncation eps (this step()'s Schwarz tol is fixed --
      // see the snapshot above)
      strategy_->update_convergence_criteria(error, global_sums[2]);

      // Release the CUDA caching allocator's unused cached blocks every
      // Schwarz iteration. AMEn's per-core local-subproblem sizes vary from
      // one iteration to the next as TT ranks adapt, which defeats the
      // caching allocator's block reuse and otherwise lets "reserved" GPU
      // memory ratchet up far past what's ever simultaneously live (see the
      // GPU-memory investigation report). Measured cost is within run-to-run
      // noise (~2% on a short benchmark) -- negligible next to the several
      // seconds each Schwarz iteration's AMEn local solves take.
      if (cfg.use_gpu && torch::cuda::is_available()) {
        torch::cuda::synchronize();
#ifdef USE_CUDA
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
      }

      // Check if the interfaces converged
      if (verbose) {
        std::cout << "-- (" << j << "): Current L2-Error = " << std::fixed
                  << std::setprecision(10) << error
                  << ", Elapsed Time = " << std::fixed << std::setprecision(3)
                  << static_cast<double>(
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count()) *
                       1e-3
                  << " s" << std::defaultfloat << std::endl;
      }
      if (error < tol) {
        break;
      }
    }

    is_converged_ = error < tol;
    if (verbose) {
      std::cout << "-- "
                << (is_converged_ ? "Converged!" : "Failed to Converge!")
                << std::endl;
    }
  }

  /// @brief Finalize the DD solver and remove any remaining information from
  /// the GPU.
  void finalize() override
  {
    if (!is_initialized_ || is_finalized_) {
      throw utils::runtime_error(*this, error_context("finalize"),
        "The solver has not been initialized yet or has "
        "already been finalized");
    }

    // Remove linear systems from GPU
    for (auto& sys : local_systems_) {
      sys->to_(torch::device(torch::kCPU));
    }

    is_finalized_ = true;
  }

  /// @return Whether the DD solver has been initialized.
  bool is_initialized() const noexcept { return is_initialized_; }
  /// @return Whether the DD solver has been finalized.
  bool is_finalized() const noexcept { return is_finalized_; }

  /// @return Whether the last step() call's Schwarz loop broke via
  /// convergence rather than hitting max_iter.
  bool is_converged() const noexcept { return is_converged_; }

  /// @return Number of Schwarz (block-Jacobi) iterations the most recent
  /// step() call ran.
  int last_num_iterations() const noexcept override final
  {
    return last_num_iterations_;
  }

  /// @return The Schwarz L2 error at each iteration of the most recent
  /// step() call, in iteration order.
  std::vector<double> last_errors() const override final
  {
    return last_errors_;
  }

  // =================================================================
  // Public getters / setters
  /// @return The label of the DD solver.
  const Label& get_label() const noexcept { return label_; }
  /// @return Get the pointer to the DD strategy.
  const DDStrategy::Ptr& get_strategy() const noexcept { return strategy_; }
  /// @return A vector of linear systems for each mesh block local to this rank.
  const Systems& get_local_systems() const override { return local_systems_; }
  /// @return GID-to-local-index map for the local systems.
  const std::unordered_map<int64_t, size_t>& get_gid_map() const noexcept
  {
    return gid_to_local_idx_;
  }
  /// @return Get the world communicator.
  const parallel::Communicator& get_world_comm() const noexcept
  {
    return world_comm_;
  }
  /// @return Get the boundary communicators.
  const parallel::BoundaryCommunicator& get_boundary_comms() const noexcept
  {
    return boundary_comms_;
  }
  /// @return Get the GPU stream pool.
  const parallel::StreamPool::Ptr get_stream_pool() const noexcept
  {
    return scheduler_.get_stream_pool();
  }
  /// @brief Set the local linear systems and rebuild the GID map.
  /// @param local_systems Systems for each mesh block on this MPI rank.
  ///        Each system must have its GID set via LinearSystem::set_gid().
  void set_local_systems(const Systems& local_systems) override
  {
    init(local_systems);
  }
  /// @brief Set a callback invoked every `frequency`-th Schwarz sweep inside
  /// step(), with this DDSolver passed by const reference -- the callback
  /// body can pull whatever it needs off it (get_local_systems(), get_eps(),
  /// last_num_iterations(), last_errors(), ...). Pass nullptr (the default)
  /// to disable. `frequency` is checked before callback_ is invoked, so a
  /// skipped sweep (e.g. a Python callback wrapped in a GIL acquire by the
  /// binding) never pays for the call at all.
  /// @param frequency Invoke callback every `frequency`-th sweep (1 = every
  /// sweep, the default). Must be >= 1.
  void set_callback(Callback callback, int frequency = 1)
  {
    if (frequency < 1) {
      throw utils::runtime_error(*this, error_context("set_callback"),
        "`frequency` must be greater than or equal to 1");
    }
    callback_ = std::move(callback);
    callback_frequency_ = frequency;
  }

  /// @return The state format of the strategy's local solver.
  linalg::FormatType get_state_format() override
  {
    return strategy_->get_local_solver()->get_state_format();
  }

  /// @return The current truncation tolerance of the strategy's local
  /// solver.
  double get_eps() const override
  {
    return strategy_->get_local_solver()->get_eps();
  }
  /// @return The maximum rank of the strategy's local solver.
  int64_t get_max_rank() const override
  {
    return strategy_->get_local_solver()->get_max_rank();
  }
};

} // namespace ttnte::solvers
