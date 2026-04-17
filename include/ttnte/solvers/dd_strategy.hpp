#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/parallel/stream_pool.hpp"
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
  void build_iteration_dag(
    task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const;
  virtual void build_cpu_iteration_dag(
    task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const;
  virtual void build_gpu_iteration_dag(
    task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const;

  // =================================================================
  // Public getters / setters
  bool use_gpu() const noexcept { return use_gpu_; }
  const MemoryPolicy& get_memory_policy() const noexcept
  {
    return memory_policy_;
  }

  void set_local_solver(const LocalSolver::Ptr& local_solver)
  {
    local_solver_ = local_solver;
  }
};

} // namespace ttnte::solvers
