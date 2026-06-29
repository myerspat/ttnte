#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/linalg/tt_config.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/boundary_communicator.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/parallel/request.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/solvers/solver_configs.hpp"
#include "ttnte/task/task_graph.hpp"
#include "ttnte/task/task_scheduler.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/label.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace ttnte::solvers {

/// @brief The domain decomposition solver class.
template<typename BlockType>
class DDSolver {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<DDSolver>;
  using Ptr = std::shared_ptr<DDSolver>;
  using Mesh = mesh::Mesh<BlockType>;

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
  /// Scheduler for executing the solver DAG.
  task::TaskScheduler scheduler_;
  /// GPU device.
  const torch::Device device_;

  /// The world communicator.
  parallel::Communicator world_comm_;
  /// The boundary communicators.
  parallel::BoundaryCommunicator boundary_comms_;
  /// The stream pool for GPU streams.
  parallel::StreamPool::Ptr stream_pool_;

  // Dynamic data during the solve.
  /// Current tolerance of the system.
  double tol_ = 1.0;
  /// The minimum error achieved by this DD solver thus far.
  double min_error_ = 1.0;

  // State variables
  bool is_initialized_ = false;
  bool is_finalized_ = false;

  // =================================================================
  // Private constructors
  DDSolver(Mesh::Ptr mesh, DDStrategy::Ptr strategy,
    std::optional<std::string> label = std::nullopt)
    : mesh_(std::move(mesh)), strategy_(std::move(strategy)),
      world_comm_(parallel::Communicator::world()),
      boundary_comms_(world_comm_, 2 * mesh_->get_ndim()),
      scheduler_(strategy_->get_config().num_threads),
      stream_pool_(
        parallel::StreamPool::instance(strategy_->get_config().num_streams)),
      device_(parallel::ParallelContext::instance().device()),
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
  void wait_for_thread_init() { scheduler_.wait_for_init(); }

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
        dag_, local_systems_, gid_to_local_idx_, boundary_comms_, stream_pool_);
    } else {
      strategy_->build_cpu_iteration_dag(
        dag_, local_systems_, gid_to_local_idx_, boundary_comms_);
    }
  }

  /// @brief Initialize the DD solver. This method will send systems to GPU
  /// depending on the MemoryPolicy.
  /// @param local_systems The vector of systems local to this MPI rank.
  void init(const std::vector<linalg::LinearSystem::Ptr>& local_systems)
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
  }

  /// @brief Run the solver.
  void step()
  {
    if (!is_initialized_ || is_finalized_) {
      throw utils::runtime_error(*this, error_context("step"),
        "The solver has not been initialized yet or has "
        "already been finalized");
    }

    double error;
    parallel::Request ereq;

    const auto& cfg = strategy_->get_config();
    bool verbose = cfg.verbose && world_comm_.rank() == 0;

    tol_ = std::max(cfg.tol, cfg.inner_forcing * min_error_);
    strategy_->update_eps(
      std::max(cfg.rounding.eps, cfg.eps_forcing * min_error_));

    torch::cuda::synchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < cfg.max_iter; j++) {
      // Execute the DAG
      scheduler_.execute(dag_);

      // Compute the total squared L2 norms for the difference between this
      // iteration and last as well as last iterations boundary solution
      double local_sums[2] = {0.0, 0.0};
      for (const auto& sys : local_systems_) {
        for (const auto& coupling : sys->get_couplings()) {
          local_sums[0] += coupling.sq_diff;
          local_sums[1] += coupling.sq_prev;
        }
      }

      // Sum the local norms with all MPI ranks
      double global_sums[2] = {local_sums[0], local_sums[1]};
      if (world_comm_.size() > 1) {
        ereq = world_comm_.iallreduce(
          local_sums, global_sums, 2, parallel::MPIOp::SUM);
      }

      // Reset the DAG
      dag_.reset();

      // Wait for MPI communication
      ereq.wait();

      // Compute error for DD iteration
      error = (global_sums[1] > 0.0)
                ? std::sqrt(global_sums[0] / global_sums[1])
                : std::numeric_limits<double>::max();

      // Tighten TT truncation eps and Schwarz inner tol independently
      if (error < min_error_ && error > 0) {
        min_error_ = error;
        strategy_->update_eps(
          std::max(cfg.rounding.eps, cfg.eps_forcing * min_error_));
      }

      // Check if the angular flux converged
      if (verbose) {
        std::cout << "-- (" << j
                  << "): Interface Flux L2-Error = " << std::fixed
                  << std::setprecision(10) << error
                  << ", Elapsed Time = " << std::fixed << std::setprecision(3)
                  << static_cast<double>(
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count()) *
                       1e-3
                  << " s" << std::defaultfloat << std::endl;
      }
      if (error < tol_) {
        break;
      }
    }

    if (verbose) {
      std::cout << "-- "
                << ((error < tol_) ? "Converged!" : "Failed to Converge!")
                << std::endl;
    }
  }

  /// @brief Finalize the DD solver and remove any remaining information from
  /// the GPU.
  void finalize()
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

  // =================================================================
  // Public getters / setters
  /// @return The label of the DD solver.
  const Label& get_label() const noexcept { return label_; }
  /// @return Get the pointer to the DD strategy.
  const DDStrategy::Ptr& get_strategy() const noexcept { return strategy_; }
  /// @return A vector of linear systems for each mesh block local to this rank.
  const std::vector<linalg::LinearSystem::Ptr>& get_local_systems()
    const noexcept
  {
    return local_systems_;
  }
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
    return stream_pool_;
  }
  /// @brief Set the local linear systems and rebuild the GID map.
  /// @param local_systems Systems for each mesh block on this MPI rank.
  ///        Each system must have its GID set via LinearSystem::set_gid().
  void set_local_systems(
    const std::vector<linalg::LinearSystem::Ptr>& local_systems)
  {
    local_systems_ = local_systems;
    gid_to_local_idx_.clear();
    for (size_t i = 0; i < local_systems_.size(); ++i) {
      gid_to_local_idx_[local_systems_[i]->get_gid()] = i;
    }
  }
};

} // namespace ttnte::solvers
