#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/solvers/memory_policy.hpp"
#include "ttnte/task/task_graph.hpp"
#include <memory>

namespace ttnte::solvers {

#ifdef USE_CUDA
inline constexpr bool DEFAULT_USE_GPU = true;
inline constexpr MemoryPolicy DEFAULT_MEMORY_POLICY = MemoryPolicy::RESIDENT;
#else
inline constexpr bool DEFAULT_USE_GPU = false;
inline constexpr MemoryPolicy DEFAULT_MEMORY_POLICY = MemoryPolicy::OUT_OF_CORE;
#endif

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
  /// Boolean for whether to use the GPU or not.
  bool use_gpu_;
  /// The memory policy for how the solver is to use GPU VRAM.
  MemoryPolicy memory_policy_;
  /// A shared pointer to the local system solver.
  LocalSolver::Ptr local_solver_;

  // =================================================================
  // Protected constructors
  DDStrategy(bool use_gpu = DEFAULT_USE_GPU,
    MemoryPolicy memory_policy = DEFAULT_MEMORY_POLICY);

public:
  virtual ~DDStrategy() = default;

  // =================================================================
  // Public methods
  /// @brief Build the iteration dag for this strategy.
  /// @param dag The task graph to be filled.
  /// @param local_systems A vector of local systems for this MPI rank.
  void build_iteration_dag(
    task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const;
  /// @brief Build the iteration dag for this strategy with no GPU support.
  /// @param dag The task graph to be filled.
  /// @param local_systems A vector of local systems for this MPI rank.
  virtual void build_cpu_iteration_dag(
    task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const;
  /// @brief Build the iteration dag for this strategy with GPU support.
  /// @param dag The task graph to be filled.
  /// @param local_systems A vector of local systems for this MPI rank.
  virtual void build_gpu_iteration_dag(
    task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const;

  // =================================================================
  // Public getters / setters
  /// @return Whether to use GPUs or not.
  bool use_gpu() const noexcept { return use_gpu_; }
  /// @return Get the memory policy used by this strategy.
  const MemoryPolicy& get_memory_policy() const noexcept
  {
    return memory_policy_;
  }

  // TODO: Change this to use multiple different types of local solvers for
  // various patches depending on some heuristic such as the maximum solution
  // rank. Therefore, if TT proves to be a difficult representation for a
  // particular part of the problem then we can just transform that to a dense
  // vector and use a different solver.

  /// @param local_solver The new local solver for this strategy.
  void set_local_solver(const LocalSolver::Ptr& local_solver)
  {
    local_solver_ = local_solver;
  }
};

} // namespace ttnte::solvers
