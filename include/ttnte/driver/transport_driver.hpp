#pragma once

#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/heuristics.hpp"
#include "ttnte/parallel/load_balancer.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include "ttnte/physics/dg_first_order_transport_assembler.hpp"
#include "ttnte/solvers/dd_solver.hpp"
#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/label.hpp"
#include "ttnte/xs/server.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>

namespace ttnte::driver {

template<typename BlockType, int64_t NumDim>
class TransportDriver {
  friend parallel::LoadBalancer<BlockType>;

public:
  // =================================================================
  // Public types
  using Mesh = mesh::Mesh<BlockType>;
  using Label = utils::Label<TransportDriver>;
  using Ptr = std::shared_ptr<TransportDriver>;
  using Assembler = physics::DGFirstOrderTransportAssembler<BlockType, NumDim>;
  using Solver = solvers::DDSolver<BlockType>;

  // Communication and load balancing
  using Communicator = parallel::Communicator;
  using LoadHeuristicPtr =
    std::shared_ptr<parallel::heuristics::LoadHeuristic<BlockType>>;
  using LoadBalancer = parallel::LoadBalancer<BlockType>;

private:
  // =================================================================
  // Private types
  /// Per-patch assembled data. assembler is null after clear_assemblers.
  struct PatchData {
    Assembler::Ptr assembler;
    linalg::LinearSystem::Ptr system;
  };

  // =================================================================
  // Private data
  /// Label
  Label label_;
  /// Shared pointer to the mesh
  Mesh::Ptr mesh_;
  /// Shared pointer to the cross section server
  xs::Server::Ptr xs_server_;

  // Communication and load balancing
  /// Communicator responsible for MPI communication
  Communicator comm_;
  /// Load balancer for running (Par)METIS
  LoadBalancer load_balancer_;

  // Per-patch assembled data (populated by assemble())
  std::unordered_map<int64_t, PatchData> patch_data_;

  // Solver (populated by solve_eigenvalue())
  Solver::Ptr solver_;

  // States
  bool is_distributed_ = false;

  // =================================================================
  // Private constructors
  TransportDriver(Mesh::Ptr mesh, xs::Server::Ptr xs_server,
    const parallel::ParallelContext& mpi_context,
    std::optional<std::string> label = std::nullopt)
    : mesh_(std::move(mesh)), xs_server_(std::move(xs_server)),
      comm_(Communicator::world()), load_balancer_(mpi_context.world_size()),
      label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {
    // Finalize the mesh and server if not already finalized
    if (!mesh_->is_finalized()) {
      mesh_->finalize();
    }
    if (!xs_server_->is_finalized()) {
      xs_server_->finalize();
    }
  }

public:
  // =================================================================
  // Public methods
  /// @brief Build a TransportDriver and get the shared pointer to it.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(
      new TransportDriver<BlockType, NumDim>(std::forward<Args>(args)...));
  }

  /// @brief Assemble the linear system for each local mesh block.
  /// @param angular_qset Angular quadrature set.
  /// @param config Assembler configuration (formats, rounding parameters,
  /// etc.).
  void assemble(const math::QuadratureSet::Ptr& angular_qset,
    const physics::DGTransportAssemblerConfig& config)
  {
    patch_data_.clear();

    for (const auto& block : mesh_->get_blocks()) {
      const int64_t gid = block->get_gid();
      auto asm_ptr = Assembler::create(block, angular_qset, xs_server_, config);
      PatchData pd;
      pd.system = asm_ptr->assemble();
      pd.system->set_gid(gid);
      pd.assembler = std::move(asm_ptr);
      patch_data_.emplace(gid, std::move(pd));
    }
  }

  /// @brief Run the eigenvalue solver using the given strategy.
  /// @param strategy Domain-decomposition strategy that owns the
  /// DDSolverConfig.
  /// @param tol The convergence tolerance on the relative Frobenius (L2) error
  /// of the angular flux.
  /// @param max_iter The maximum number of outer iterations.
  /// @param verbose Whether to print outer iteration progress.
  /// @return The resulting eigenvalue of the linear system.
  double solve_eigenvalue(solvers::DDStrategy::Ptr strategy, double tol = 1e-8,
    int max_iter = 500, bool verbose = true)
  {
    if (patch_data_.empty()) {
      throw utils::runtime_error(
        "ttnte::driver::TransportDriver::solve_eigenvalue",
        "No linear systems assembled. Call assemble() first.");
    }

    // Initialize the solver
    double k_global = init_solver(std::move(strategy));
    assert(solver_->is_initialized() && !solver_->is_finalized());

    // Ensure all worker threads have set their CUDA device before the first
    // iteration so cuBLAS context initialization does not produce warnings.
    solver_->wait_for_thread_init();

    parallel::Request ereq;
    parallel::Request kreq;
    const double one = 1.0;
    const auto& cfg = solver_->get_strategy()->get_config();
    const auto& local_systems = solver_->get_local_systems();
    verbose = verbose && comm_.rank() == 0;

    double error = std::numeric_limits<double>::max();
    double min_error = 0.1;

    // Previous patch solutions for computing errors
    std::vector<linalg::State> states(local_systems.size());

    // Begin transport iteration
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_iter; i++) {
      // Snapshot flux before the inner Schwarz so we can measure outer change
      for (size_t idx = 0; idx < local_systems.size(); ++idx) {
        states[idx] = local_systems[idx]->get_state();
      }

      // Set eigenvalues of the multiplying systems
      for (const auto& sys : local_systems) {
        const auto& src = sys->get_source();
        if (src && src->is_eigenvalue()) {
          auto fsrc = std::static_pointer_cast<linalg::EigenSource>(src);
          fsrc->set_eigval(one / k_global);
          fsrc->scale();
        }
      }

      // Run the DD solver
      solver_->step();

      // Compute the local error in the eigenvector using a relative L2 error to
      // the last iteration
      double local_outer_sums[2] = {0.0, 0.0};
      for (size_t idx = 0; idx < local_systems.size(); ++idx) {
        const linalg::State& prev = states[idx];
        if (!prev.defined())
          continue;

        linalg::State diff = local_systems[idx]->get_state() - prev;
        const double n_diff = diff.norm();
        const double n_prev = prev.norm();
        local_outer_sums[0] += n_diff * n_diff;
        local_outer_sums[1] += n_prev * n_prev;
      }

      // Sum the errors across all MPI ranks
      double global_outer_sums[2] = {local_outer_sums[0], local_outer_sums[1]};
      if (comm_.size() > 1) {
        ereq = comm_.iallreduce(
          local_outer_sums, global_outer_sums, 2, parallel::MPIOp::SUM);
      }

      // Calculate the local fission source for this MPI rank
      double k = 0.0;
      for (const auto& sys : local_systems) {
        const auto& src = sys->get_source();
        if (src && src->is_eigenvalue()) {
          auto fsrc = std::static_pointer_cast<linalg::EigenSource>(src);
          fsrc->update(
            sys->get_state(), cfg.rounding.eps, cfg.rounding.max_rank);
          k += fsrc->get_total_source();
        }
      }

      // Sum the fission sources across all MPI ranks
      if (comm_.size() > 1) {
        kreq = comm_.iallreduce(&k, &k_global, 1, parallel::MPIOp::SUM);
      } else {
        k_global = k;
      }

      // Wait for allreduce and update outer iteration errors
      ereq.wait();
      error = (global_outer_sums[1] > 0.0)
                ? std::sqrt(global_outer_sums[0] / global_outer_sums[1])
                : std::numeric_limits<double>::max();

      min_error = std::min(error, min_error);
      kreq.wait();

      if (verbose) {
        std::cout << "-- (" << i << "): k = " << std::fixed
                  << std::setprecision(6) << k_global
                  << ", Angular Flux L2 Error = " << std::fixed
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

    // Remove the linear systems from GPU
    solver_->finalize();

    if (verbose) {
      std::cout << "-- "
                << ((error < tol) ? "Converged!" : "Failed to Converge!")
                << std::endl;
    }

    for (const auto& sys : local_systems) {
      const auto& x = sys->get_state();
      std::cout << "Ranks: " << x.as_tt().get_ranks()
                << " Compression: " << x.get_compression() << std::endl;
    }

    // TODO: create a result class or something that takes the eigenvalue and
    // angular flux states or just all the linear systems. This class should
    // probably be capable of plotting and what not for post solve stuff.
    return k_global;
  }

  /// @brief Partition the Mesh according to the load heuristics. Partitioning
  /// runs METIS_PartGraphKway on rank root_rank.
  /// @param load_heuristics These heuristics compute weights for each local
  /// MeshBlock to help METIS decide the best repartition.
  /// @param root_rank The rank to run METIS on.
  void distribute(
    std::vector<LoadHeuristicPtr> load_heuristics = {}, int root_rank = 0)
  {
    // MPI world size is one (only one MPI rank)
    if (comm_.size() == 1) {
      return;
    }

    // Finalize the heuristics
    for (auto& heuristic : load_heuristics) {
      heuristic->finalize(mesh_);
    }

    // Determine what MeshBlocks will stay on this MPI rank
    auto [local_gids, gid2rank] = load_balancer_.compute_partition(
      std::move(mesh_->build_connectivity_graph()), comm_, load_heuristics,
      root_rank);

    // Restrict the mesh
    mesh_->cull_blocks(std::move(local_gids), std::move(gid2rank));
    is_distributed_ = true;
  }

  /// @brief Repartition the Mesh according to the load heuristics.
  /// Repartitioning runs ParMETIS_V3_AdaptiveRepart
  /// (https://karypis.github.io/glaros/files/sw/parmetis/manual.pdf).
  /// @param load_heuristics These heuristics compute weights for each local
  /// MeshBlock to help ParMETIS decide the best repartition.
  void redistribute(std::vector<LoadHeuristicPtr> load_heuristics = {})
  {
    // MPI world size is one (only one MPI rank)
    if (comm_.size() == 1) {
      return;
    }

    // Finalize the heuristics
    for (auto& heuristic : load_heuristics) {
      heuristic->finalize(mesh_);
    }

    // Create routing table for sending MeshBlocks around
    const auto& routing_table = load_balancer_.compute_repartition(
      std::move(mesh_->build_connectivity_graph()), comm_, load_heuristics);
  }

  /// @brief Initialize the DD solver.
  /// @param strategy The DD strategy.
  /// @return The global fission source.
  double init_solver(solvers::DDStrategy::Ptr strategy)
  {
    // Clear assemblers
    if (strategy->get_config().clear_assemblers) {
      for (auto& [gid, pd] : patch_data_) {
        pd.assembler = nullptr;
      }
    }

    // Build the local systems vector in mesh block order
    std::vector<linalg::LinearSystem::Ptr> local_systems;
    local_systems.reserve(patch_data_.size());
    for (const auto& block : mesh_->get_blocks()) {
      local_systems.push_back(patch_data_.at(block->get_gid()).system);
    }

    // Get config information
    const auto& cfg = strategy->get_config();

    // Populate the initial guess and compute the initial fission source
    // (k-eigenvalue)
    double k = 0;
    for (auto& sys : local_systems) {
      const auto& interior_op = sys->get_interior_op();
      const auto& n_modes = interior_op.as_tt().get_n_modes();
      const auto& device = interior_op.get_device();
      const auto& dtype = interior_op.get_dtype();

      // Check if this is a fissile system
      linalg::State psi = linalg::State::ones(cfg.fmt, n_modes, device, dtype);
      const auto& src = sys->get_source();
      if (src && src->is_eigenvalue()) {
        // Update the fission source
        const auto& fsrc = std::static_pointer_cast<linalg::EigenSource>(src);
        fsrc->update(psi, cfg.rounding.eps, cfg.rounding.max_rank);

        // Compute this patch's contribution
        k += fsrc->get_total_source();
      }

      // for (auto& coupling : sys->get_couplings()) {
      //   auto face = linalg::State::ones(
      //     cfg.fmt, coupling.boundary_op.as_tt().get_n_modes(), device,
      //     dtype);
      //   coupling.recv_buffer = std::move(face);
      // }

      // Set the initial guess
      sys->set_state(std::move(psi));
    }

    // Determine the total fission source (k-eigenvalue) across all ranks
    parallel::Request kreq;
    double k_global = k;
    if (comm_.size() > 1) {
      kreq = comm_.iallreduce(&k, &k_global, 1, parallel::MPIOp::SUM);
    }

    // Create the DD solver and register all local systems
    solver_ = Solver::create(mesh_, std::move(strategy));

    // Initialize the solver
    solver_->init(std::move(local_systems));

    // Construct the solver DAG
    solver_->build_iteration_dag();

    // Wait for the eigenvalue to be sent across ranks
    kreq.wait();
    return k_global;
  }

  // =================================================================
  // Public getters / setters
  /// @return Get the label of the driver.
  const Label& get_label() const noexcept { return label_; }
  /// @return Get the shared pointer to the mesh.
  const Mesh::Ptr& get_mesh() const noexcept { return mesh_; }
  /// @return Get the shared pointer to the cross section library.
  const xs::Server::Ptr& get_server() const noexcept { return xs_server_; }
  /// @return The DD solver created by the last solve_eigenvalue() call, or
  /// null.
  const Solver::Ptr& get_solver() const noexcept { return solver_; }

  /// @brief Get the assembler for a specific mesh block GID.
  /// @param gid Global ID of the mesh block.
  /// @throws runtime_error if the assembler was cleared or GID is unknown.
  const Assembler::Ptr& get_assembler(int64_t gid) const
  {
    auto it = patch_data_.find(gid);
    if (it == patch_data_.end()) {
      throw utils::runtime_error(
        "ttnte::driver::TransportDriver::get_assembler",
        "No patch data found for GID " + std::to_string(gid) +
          ". Has assemble() been called?");
    }
    if (!it->second.assembler) {
      throw utils::runtime_error(
        "ttnte::driver::TransportDriver::get_assembler",
        "Assembler for GID " + std::to_string(gid) +
          " was cleared during solve. Set clear_assemblers=false "
          "in DDSolverConfig to retain it.");
    }
    return it->second.assembler;
  }

  /// @brief Get the assembled linear system for a specific mesh block GID.
  /// @param gid Global ID of the mesh block.
  /// @throws runtime_error if GID is unknown or assemble() has not been called.
  const linalg::LinearSystem::Ptr& get_system(int64_t gid) const
  {
    auto it = patch_data_.find(gid);
    if (it == patch_data_.end()) {
      throw utils::runtime_error("ttnte::driver::TransportDriver::get_system",
        "No patch data found for GID " + std::to_string(gid) +
          ". Has assemble() been called?");
    }
    return it->second.system;
  }

  /// @param label The new label of the driver.
  void set_label(const std::string& label)
  {
    label_ = Label::from_string(label);
  }
};

} // namespace ttnte::driver
