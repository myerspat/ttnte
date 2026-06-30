#include "ttnte/solvers/block_jacobi_strategy.hpp"
#include "ttnte/task/configure_cpu_task.hpp"
#include "ttnte/task/configure_cuda_task.hpp"
#include "ttnte/task/configure_mpi_task.hpp"
#include "ttnte/utils/exception.hpp"
#include <ATen/core/interned_strings.h>
#include <unordered_map>

namespace {

void build_iteration_dag(ttnte::task::TaskGraph& dag,
  const ttnte::solvers::BlockJacobiStrategy& strategy,
  const std::vector<ttnte::linalg::LinearSystem::Ptr>& local_systems,
  const std::unordered_map<int64_t, size_t>& gid_to_local,
  const ttnte::parallel::BoundaryCommunicator& boundary_comms,
  const ttnte::parallel::StreamPool::Ptr& stream_pool = nullptr)
{
  int my_rank = boundary_comms.get_comms()[0].rank();
  const auto& config = strategy.get_config();
  const bool use_gpu_recv =
    config.use_gpu &&
    (config.memory_policy == ttnte::solvers::MemoryPolicy::RESIDENT ||
      config.memory_policy == ttnte::solvers::MemoryPolicy::STATE_RESIDENT);
  const torch::Device device =
    use_gpu_recv ? stream_pool->get_device() : torch::Device(torch::kCPU);
  const auto comm_device = config.comm_mode == ttnte::solvers::CommMode::SYNC
                             ? ttnte::task::DeviceTarget::CPU_SYNC
                             : ttnte::task::DeviceTarget::CPU_ASYNC;

  // Iterate through the local linear systems
  std::vector<c10::SmallVector<ttnte::task::Task*, 6>> compute_tasks;
  compute_tasks.reserve(local_systems.size());
  for (const auto& sys : local_systems) {
    int64_t gid = sys->get_gid();
    auto& couplings = sys->get_couplings();

    // Create dynamic non-blocking receive and unpack tasks
    c10::SmallVector<ttnte::task::Task*, 6> unpack_tasks(couplings.size());
    for (size_t i = 0; i < couplings.size(); i++) {
      auto& coupling = couplings[i];

      if (my_rank != coupling.connection.mpi_rank) {
        // Create receive buffer on the appropriate device
        auto recv_buffer = std::make_shared<torch::Tensor>(torch::empty(
          {0}, torch::TensorOptions().dtype(sys->get_dtype()).device(device)));

        // Get global ID for this system
        int64_t sgid = coupling.connection.gid;

        // Generate a tag
        auto tag = boundary_comms.generate_tag(sgid, 0, gid, 0);

        // Create non-blocking dynamic receive task
        auto irecv_task = dag.create_task(comm_device,
          "irecv_" + std::to_string(sgid) + "to" + std::to_string(gid));
        ttnte::task::mpi::configure_dynamic_irecv_task(*irecv_task,
          recv_buffer.get(), coupling.connection.mpi_rank, tag,
          boundary_comms.get_comm(coupling.connection.fid), device);

        // Create the unpack task
        auto unpack_task = dag.create_task(comm_device,
          "unpack_" + std::to_string(sgid) + "to" + std::to_string(gid));
        ttnte::task::cpu::configure_unpack_task(
          *unpack_task, &coupling, std::move(recv_buffer));
        unpack_task->add_dependency(irecv_task);
        unpack_tasks[i] = std::move(unpack_task);
      }
    }

    // Create compute task for this linear system
    c10::SmallVector<ttnte::task::Task*, 6> out_tasks;
    if (config.use_gpu) {
      out_tasks = strategy.build_gpu_compute_dag(dag, sys, stream_pool);
    } else {
      out_tasks = strategy.build_cpu_compute_dag(dag, sys);
    }

    // Handle MPI outflow communication
    for (size_t i = 0; i < couplings.size(); i++) {
      auto& coupling = couplings[i];
      auto& out_task = out_tasks[i];

      if (my_rank != coupling.connection.mpi_rank) {
        // Add the out task as a dependency of unpack
        unpack_tasks[i]->add_dependency(out_task);

        // Create send buffer
        auto send_buffer = std::make_shared<torch::Tensor>();

        // Get target global ID
        int64_t tgid = coupling.connection.gid;

        // Create pack and dynamic non-blocking MPI send tasks
        auto pack_task = dag.create_task(comm_device,
          "pack_" + std::to_string(gid) + "to" + std::to_string(tgid));
        ttnte::task::cpu::configure_pack_task(
          *pack_task, &coupling, send_buffer);
        pack_task->add_dependency(out_task);

        // Generate a tag
        auto tag = boundary_comms.generate_tag(gid, 0, tgid, 0);

        auto isend_task = dag.create_task(comm_device,
          "isend_" + std::to_string(gid) + "to" + std::to_string(tgid));
        ttnte::task::mpi::configure_dynamic_isend_task(*isend_task,
          send_buffer.get(), coupling.connection.mpi_rank, tag,
          boundary_comms.get_comm(coupling.fid));
        isend_task->add_dependency(pack_task);
      }
    }

    // Move out tasks to compute task map
    compute_tasks.push_back(std::move(out_tasks));
  }

  // Iterate and find local linear systems and handle local communication
  for (size_t i = 0; i < local_systems.size(); i++) {
    // Source mesh block info
    const auto& ssys = local_systems[i];
    int64_t sgid = ssys->get_gid();
    auto& scouplings = ssys->get_couplings();
    const auto& stasks = compute_tasks[i];

    for (size_t j = 0; j < scouplings.size(); j++) {
      // Source mesh block info for coupling j
      auto& scoupling = scouplings[j];
      auto stask = stasks[j];
      int64_t sfid = scoupling.fid;

      if (my_rank == scoupling.connection.mpi_rank) {
        // Target mesh block info for coupling j
        int64_t tgid = scoupling.connection.gid;
        size_t idx = gid_to_local.at(tgid);
        auto& tsys = local_systems[idx];
        auto& tcouplings = tsys->get_couplings();
        const auto& ttasks = compute_tasks[idx];

        ttnte::linalg::NeighborCoupling* tcoupling = nullptr;
        ttnte::task::Task* ttask = nullptr;
        for (size_t k = 0; k < tcouplings.size(); k++) {
          if (sgid == tcouplings[k].connection.gid &&
              sfid == tcouplings[k].connection.fid) {
            tcoupling = &tcouplings[k];
            ttask = ttasks[k];
            break;
          }
        }

        if (!tcoupling || !ttask) {
          throw ttnte::utils::runtime_error(
            "ttnte::solvers::BlockJacobiStrategy::build_gpu_iteration_dag",
            "Connection at an interface boundary not found on the same rank");
        }

        auto send_task = dag.create_task(
          comm_device, "within_rank_send_" + std::to_string(tgid) + "to" +
                         std::to_string(sgid));
        ttnte::task::cpu::configure_within_rank_send(
          *send_task, tcoupling, &scoupling);
        send_task->add_dependency(stask);
        send_task->add_dependency(ttask);
      }
    }
  }
}

} // namespace

namespace ttnte::solvers {

// =================================================================
// Public methods
void BlockJacobiStrategy::build_cpu_iteration_dag(task::TaskGraph& dag,
  const std::vector<SystemPtr>& local_systems,
  const std::unordered_map<int64_t, size_t>& gid_to_local,
  const parallel::BoundaryCommunicator& boundary_comms) const
{
  build_iteration_dag(dag, *this, local_systems, gid_to_local, boundary_comms);
}

void BlockJacobiStrategy::build_gpu_iteration_dag(task::TaskGraph& dag,
  const std::vector<linalg::LinearSystem::Ptr>& local_systems,
  const std::unordered_map<int64_t, size_t>& gid_to_local,
  const parallel::BoundaryCommunicator& boundary_comms,
  const parallel::StreamPool::Ptr& stream_pool) const
{
  build_iteration_dag(
    dag, *this, local_systems, gid_to_local, boundary_comms, stream_pool);
}

c10::SmallVector<task::Task*, 6> BlockJacobiStrategy::build_cpu_compute_dag(
  task::TaskGraph& dag, const SystemPtr& local_system) const
{
  int64_t num_couplings = local_system->get_couplings().size();
  int64_t gid = local_system->get_gid();
  const task::DeviceTarget exec_device = config_.exec_mode == ExecMode::SYNC
                                           ? task::DeviceTarget::CPU_SYNC
                                           : task::DeviceTarget::CPU_ASYNC;

  // Boundary operator apply tasks
  c10::SmallVector<task::Task*, 6> apply_tasks;
  apply_tasks.reserve(num_couplings);
  for (auto& coupling : local_system->get_couplings()) {
    int64_t sgid = coupling.connection.gid;

    auto apply_task = dag.create_task(exec_device,
      "apply_" + std::to_string(sgid) + "to" + std::to_string(gid));
    task::cpu::configure_apply_task(*apply_task, &coupling, tt_config_);

    apply_tasks.push_back(std::move(apply_task));
  }

  // Configure solve task for this mesh block
  auto solve_task =
    dag.create_task(exec_device, "solve_" + std::to_string(gid));
  task::cpu::configure_solve_task(*solve_task, local_system, local_solver_);

  // Add apply task dependencies
  for (const auto& task : apply_tasks) {
    solve_task->add_dependency(task);
  }

  // Extract the shared boundary for each coupled face after solve
  c10::SmallVector<task::Task*, 6> narrow_tasks;
  narrow_tasks.reserve(num_couplings);
  for (auto& coupling : local_system->get_couplings()) {
    int64_t tgid = coupling.connection.gid;
    auto narrow_task = dag.create_task(exec_device,
      "narrow_" + std::to_string(gid) + "to" + std::to_string(tgid));
    task::cpu::configure_narrow_task(
      *narrow_task, local_system, &coupling, tt_config_);
    narrow_task->add_dependency(solve_task);
    narrow_tasks.push_back(std::move(narrow_task));
  }

  return narrow_tasks;
}

c10::SmallVector<task::Task*, 6> BlockJacobiStrategy::build_gpu_compute_dag(
  task::TaskGraph& dag, const SystemPtr& local_system,
  const parallel::StreamPool::Ptr& stream_pool) const
{
  int64_t num_couplings = local_system->get_couplings().size();
  int64_t gid = local_system->get_gid();
  const task::DeviceTarget exec_device = config_.exec_mode == ExecMode::SYNC
                                           ? task::DeviceTarget::GPU_SYNC
                                           : task::DeviceTarget::GPU_ASYNC;
  const task::DeviceTarget comm_device = config_.comm_mode == CommMode::SYNC
                                           ? task::DeviceTarget::GPU_SYNC
                                           : task::DeviceTarget::GPU_ASYNC;

  // Host to device send and back tasks
  task::Task* h2d_task = nullptr;
  task::Task* d2h_task = nullptr;
  if (config_.memory_policy != MemoryPolicy::RESIDENT) {
    h2d_task = dag.create_task(comm_device, "h2d_" + std::to_string(gid));
    d2h_task = dag.create_task(comm_device, "d2h_" + std::to_string(gid));

    if (config_.memory_policy == MemoryPolicy::STATE_RESIDENT) {
      // Only communicate the buffer information
      task::cuda::configure_transfer_buffer_task(
        *h2d_task, local_system, torch::Device(torch::kCUDA), stream_pool);
      task::cuda::configure_transfer_buffer_task(
        *d2h_task, local_system, torch::Device(torch::kCPU), stream_pool);

    } else if (config_.memory_policy == MemoryPolicy::OPERATOR_RESIDENT) {
      // Only communicate the non-buffer information
      task::cuda::configure_transfer_nonbuffer_task(
        *h2d_task, local_system, torch::Device(torch::kCUDA), stream_pool);
      task::cuda::configure_transfer_nonbuffer_task(
        *d2h_task, local_system, torch::Device(torch::kCPU), stream_pool);

    } else {
      // Both buffer and non-buffer are communicated each iteration
      task::cuda::configure_transfer_task(
        *h2d_task, local_system, torch::Device(torch::kCUDA), stream_pool);
      task::cuda::configure_transfer_task(
        *d2h_task, local_system, torch::Device(torch::kCPU), stream_pool);
    }
  }

  // Boundary operator apply tasks
  c10::SmallVector<task::Task*, 6> apply_tasks;
  apply_tasks.reserve(num_couplings);
  for (auto& coupling : local_system->get_couplings()) {
    int64_t sgid = coupling.connection.gid;
    auto apply_task = dag.create_task(exec_device,
      "apply_" + std::to_string(sgid) + "to" + std::to_string(gid));
    task::cuda::configure_apply_task(
      *apply_task, &coupling, stream_pool, tt_config_);

    // Add dependency
    if (h2d_task) {
      apply_task->add_dependency(h2d_task);
    }
    apply_tasks.push_back(std::move(apply_task));
  }

  // Configure solve task for this mesh block
  auto solve_task =
    dag.create_task(exec_device, "solve_" + std::to_string(gid));
  task::cuda::configure_solve_task(
    *solve_task, local_system, local_solver_, stream_pool);

  // Add transfer task as a dependency
  if (h2d_task) {
    solve_task->add_dependency(h2d_task);
  }

  // Add apply task dependencies
  for (const auto& task : apply_tasks) {
    solve_task->add_dependency(task);
  }

  // Extract the shared boundary for each coupled face after solve
  c10::SmallVector<task::Task*, 6> narrow_tasks;
  narrow_tasks.reserve(num_couplings);
  for (auto& coupling : local_system->get_couplings()) {
    int64_t tgid = coupling.connection.gid;
    auto narrow_task = dag.create_task(exec_device,
      "narrow_" + std::to_string(gid) + "to" + std::to_string(tgid));
    task::cuda::configure_narrow_task(
      *narrow_task, local_system, &coupling, stream_pool, tt_config_);
    narrow_task->add_dependency(solve_task);
    narrow_tasks.push_back(std::move(narrow_task));
  }

  // Remove from GPU
  if (d2h_task) {
    d2h_task->add_dependency(solve_task);

    for (const auto& narrow_task : narrow_tasks) {
      d2h_task->add_dependency(narrow_task);
    }
  }

  // Return the in nodes and out nodes of this branch of the DAG
  if (config_.memory_policy != MemoryPolicy::RESIDENT) {
    return c10::SmallVector<task::Task*, 6>(num_couplings, d2h_task);
  } else {
    return narrow_tasks;
  }
}

} // namespace ttnte::solvers
