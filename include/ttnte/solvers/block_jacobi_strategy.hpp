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
  BlockJacobiStrategy(DDSolverConfig config = {})
    : DDStrategy(std::move(config))
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
  /// @brief Build the iteration DAG for one block-Jacobi sweep (CPU path).
  /// @param dag The DAG to add to.
  /// @param local_system The local linear system.
  /// @param gid2local The map between the global linear system ID to the rank
  /// local one.
  /// @param boundary_comms The boundary communicators.
  void build_cpu_iteration_dag(task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems,
    const std::unordered_map<int64_t, size_t>& gid2local,
    const parallel::BoundaryCommunicator& boundary_comms) const override final;

  /// @brief Build the iteration DAG for one block-Jacobi sweep (GPU path).
  /// @param dag The DAG to add to.
  /// @param local_system The local linear system.
  /// @param gid2local The map between the global linear system ID to the rank
  /// local one.
  /// @param boundary_comms The boundary communicators.
  void build_gpu_iteration_dag(task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems,
    const std::unordered_map<int64_t, size_t>& gid2local,
    const parallel::BoundaryCommunicator& boundary_comms,
    const parallel::StreamPool::Ptr& stream_pool) const override final;

  /// @brief Add the task to solve the linear system for a certain mesh block as
  /// a CPU-based task.
  /// @param dag The dag to add the compute task to.
  /// @param local_system The local linear system.
  /// @param is_async Whether to make the compute task synchronous or
  /// asynchronous.
  /// @return The last nodes of the new portion of the DAG (one for each
  /// coupling).
  c10::SmallVector<task::Task*, 6> build_cpu_compute_dag(
    task::TaskGraph& dag, const SystemPtr& local_system) const;
  /// @brief Add the task to solve the linear system for a certain mesh block as
  /// a GPU-based task.
  /// @param dag The dag to add the compute task to.
  /// @param local_system The local linear system.
  /// @param is_async Whether to make the compute task synchronous or
  /// asynchronous.
  /// @return The last nodes of the new portion of the DAG (one for each
  /// coupling).
  c10::SmallVector<task::Task*, 6> build_gpu_compute_dag(task::TaskGraph& dag,
    const SystemPtr& local_system,
    const parallel::StreamPool::Ptr& stream_pool) const;
};

} // namespace ttnte::solvers
