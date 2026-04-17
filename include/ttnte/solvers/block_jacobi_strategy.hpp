#pragma once

#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/task/task.hpp"
#include <memory>

namespace ttnte::solvers {

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
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new BlockJacobiStrategy(std::forward<Args>(args)...));
  }

  // =================================================================
  // Public methods
  // void build_cpu_iteration_dag(task::TaskGraph& dag,
  //   const std::vector<SystemPtr>& local_systems) const override final;
  // void build_gpu_iteration_dag(task::TaskGraph& dag,
  //   const std::vector<SystemPtr>& local_systems) const override final;

  task::Task* build_cpu_compute_dag(
    task::TaskGraph& dag, const SystemPtr& local_system, bool is_async = true);
  std::tuple<task::Task*, task::Task*, task::Task*> build_gpu_compute_dag(
    task::TaskGraph& dag, const SystemPtr& local_system,
    const parallel::StreamPool::Ptr& stream_pool, bool is_async = true);
};

} // namespace ttnte::solvers
