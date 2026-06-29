#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/parallel/boundary_communicator.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/solvers/solver_configs.hpp"
#include "ttnte/task/task_graph.hpp"
#include <memory>
#include <unordered_map>

namespace ttnte::solvers {

/// @brief Strategy class for the domain decomposition solver.
class DDStrategy {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<DDStrategy>;
  using SystemPtr = std::shared_ptr<linalg::LinearSystem>;

protected:
  // =================================================================
  // Protected data
  /// A shared pointer to the local system solver.
  LocalSolver::Ptr local_solver_;
  /// Solver configuration (convergence tolerances, rounding, etc.).
  DDSolverConfig config_;
  /// Dynamic low-rank tensor network configuration.
  std::shared_ptr<linalg::TTConfig> tt_config_;

  // =================================================================
  // Protected constructors
  DDStrategy(DDSolverConfig config = {});

public:
  virtual ~DDStrategy() = default;

  // =================================================================
  // Public methods
  /// @brief Build the iteration DAG for this strategy (CPU path).
  /// Called by DDSolver::build_iteration_dag when use_gpu() is false.
  /// @param dag             Task graph to populate.
  /// @param local_systems   Systems local to this MPI rank.
  /// @param gid_to_local    GID → local_systems index map.
  /// @param boundary_comms  Per-face MPI communicators.
  /// @param boundary_cfg    Rounding config for boundary communication tasks.
  virtual void build_cpu_iteration_dag(task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems,
    const std::unordered_map<int64_t, size_t>& gid_to_local,
    const parallel::BoundaryCommunicator& boundary_comms) const;

  /// @brief Build the iteration DAG for this strategy (GPU path).
  /// Called by DDSolver::build_iteration_dag when use_gpu() is true.
  /// @param dag             Task graph to populate.
  /// @param local_systems   Systems local to this MPI rank.
  /// @param gid_to_local    GID → local_systems index map.
  /// @param boundary_comms  Per-face MPI communicators.
  /// @param stream_pool     GPU stream pool.
  virtual void build_gpu_iteration_dag(task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems,
    const std::unordered_map<int64_t, size_t>& gid_to_local,
    const parallel::BoundaryCommunicator& boundary_comms,
    const parallel::StreamPool::Ptr& stream_pool) const;

  /// @brief Update the truncation tolerance for dynamic tolerance and to avoid
  /// over-solving.
  /// @param eps The new truncation tolerance.
  void update_eps(double eps)
  {
    local_solver_->set_eps(eps);
    tt_config_->eps = eps;
  }

  // =================================================================
  // Public getters / setters
  /// @return The solver configuration (convergence tolerances, rounding, etc.).
  const DDSolverConfig& get_config() const noexcept { return config_; }
  /// @param config The new solver configuration.
  void set_config(const DDSolverConfig& config) { config_ = config; }

  // TODO: Change this to use multiple different types of local solvers for
  // various patches depending on some heuristic such as the maximum solution
  // rank. Therefore, if TT proves to be a difficult representation for a
  // particular part of the problem then we can just transform that to a dense
  // vector and use a different solver.

  /// @return The current local linear solver.
  const LocalSolver::Ptr& get_local_solver() const noexcept
  {
    return local_solver_;
  }
  /// @param local_solver The new local solver for this strategy.
  void set_local_solver(const LocalSolver::Ptr& local_solver)
  {
    local_solver_ = local_solver;
  }
};

} // namespace ttnte::solvers
