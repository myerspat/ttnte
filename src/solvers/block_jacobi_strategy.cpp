#include "ttnte/solvers/block_jacobi_strategy.hpp"
#include "ttnte/task/configure_cpu_task.hpp"
#include "ttnte/task/configure_cuda_task.hpp"

namespace ttnte::solvers {

// =================================================================
// Public methods
// void BlockJacobiStrategy::build_cpu_iteration_dag(
//   task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const
// {}
//
// void BlockJacobiStrategy::build_gpu_iteration_dag(
//   task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const
// {}

task::Task* BlockJacobiStrategy::build_cpu_compute_dag(
  task::TaskGraph& dag, const SystemPtr& local_system, bool is_async)
{
  auto solve_task = dag.create_task(
    is_async ? task::DeviceTarget::CPU_ASYNC : task::DeviceTarget::CPU_SYNC);
  task::cpu::configure_solve_task(*solve_task, local_system, local_solver_);
  return solve_task;
}

std::tuple<task::Task*, task::Task*, task::Task*>
BlockJacobiStrategy::build_gpu_compute_dag(task::TaskGraph& dag,
  const SystemPtr& local_system, const parallel::StreamPool::Ptr& stream_pool,
  bool is_async)
{
  const task::DeviceTarget target_device =
    is_async ? task::DeviceTarget::GPU_ASYNC : task::DeviceTarget::GPU_SYNC;

  // Solve task
  auto solve_task = dag.create_task(target_device);
  task::cuda::configure_solve_task(
    *solve_task, local_system, local_solver_, stream_pool);

  // Host to device send and back
  task::Task* h2d_task = nullptr;
  task::Task* d2h_task = nullptr;
  if (memory_policy_ != MemoryPolicy::RESIDENT) {
    if (memory_policy_ == MemoryPolicy::STATE_RESIDENT) {
      // Only communicate the buffer information
      h2d_task = dag.create_task(target_device);
      task::cuda::configure_transfer_buffer_task(
        *h2d_task, local_system, torch::Device(torch::kCUDA), stream_pool);
      d2h_task = dag.create_task(target_device);
      task::cuda::configure_transfer_buffer_task(
        *d2h_task, local_system, torch::Device(torch::kCPU), stream_pool);

    } else if (memory_policy_ == MemoryPolicy::OPERATOR_RESIDENT) {
      // Only communicate the non-buffer information
      h2d_task = dag.create_task(target_device);
      task::cuda::configure_transfer_nonbuffer_task(
        *h2d_task, local_system, torch::Device(torch::kCUDA), stream_pool);
      d2h_task = dag.create_task(target_device);
      task::cuda::configure_transfer_nonbuffer_task(
        *d2h_task, local_system, torch::Device(torch::kCPU), stream_pool);

    } else {
      // Both buffer and non-buffer are communicated each iteration
      h2d_task = dag.create_task(target_device);
      task::cuda::configure_transfer_task(
        *h2d_task, local_system, torch::Device(torch::kCUDA), stream_pool);
      d2h_task = dag.create_task(target_device);
      task::cuda::configure_transfer_task(
        *d2h_task, local_system, torch::Device(torch::kCPU), stream_pool);
    }

    // Add dependencies
    solve_task->add_dependency(h2d_task);
    d2h_task->add_dependency(solve_task);
  }

  return std::make_tuple(h2d_task, solve_task, d2h_task);
}

} // namespace ttnte::solvers
