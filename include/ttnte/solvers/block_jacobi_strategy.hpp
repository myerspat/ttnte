#pragma once

#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/task/task.hpp"
#include <memory>

namespace ttnte::solvers {

/// @brief The block-Jacobi domain decomposition strategy. Solve the boundary
/// trace using block-Jacobi iteration.
class BlockJacobiStrategy : public DDStrategy {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<BlockJacobiStrategy>;

protected:
  // =================================================================
  // Protected constructors
  BlockJacobiStrategy(bool use_gpu = DEFAULT_USE_GPU,
    MemoryPolicy memory_policy = DEFAULT_MEMORY_POLICY)
    : DDStrategy(use_gpu, memory_policy)
  {}

public:
  virtual ~BlockJacobiStrategy() = default;

  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new BlockJacobiStrategy instance.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new BlockJacobiStrategy(std::forward<Args>(args)...));
  }

  // =================================================================
  // Public methods
  // /// @brief Build the iteration dag for this strategy with no GPU support.
  // /// @param dag The task graph to be filled.
  // /// @param local_systems A vector of local systems for this MPI rank.
  // void build_cpu_iteration_dag(task::TaskGraph& dag,
  //   const std::vector<SystemPtr>& local_systems) const override final;
  // /// @brief Build the iteration dag for this strategy with GPU support.
  // /// @param dag The task graph to be filled.
  // /// @param local_systems A vector of local systems for this MPI rank.
  // void build_gpu_iteration_dag(task::TaskGraph& dag,
  //   const std::vector<SystemPtr>& local_systems) const override final;

  /// @brief Add the task to solve the linear system for a certain mesh block as
  /// a CPU-based task.
  /// @param dag The dag to add the compute task to.
  /// @param local_system The local linear system.
  /// @param is_async Whether to make the compute task synchronous or
  /// asynchronous.
  task::Task* build_cpu_compute_dag(
    task::TaskGraph& dag, const SystemPtr& local_system, bool is_async = true);
  /// @brief Add the task to solve the linear system for a certain mesh block as
  /// a GPU-based task.
  /// @param dag The dag to add the compute task to.
  /// @param local_system The local linear system.
  /// @param is_async Whether to make the compute task synchronous or
  /// asynchronous.
  std::tuple<task::Task*, task::Task*, task::Task*> build_gpu_compute_dag(
    task::TaskGraph& dag, const SystemPtr& local_system,
    const parallel::StreamPool::Ptr& stream_pool, bool is_async = true);
};

} // namespace ttnte::solvers
