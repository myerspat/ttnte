#pragma once

#include "ttnte/linalg/neighbor_coupling.hpp"
#include "ttnte/linalg/ops.hpp"
#include "ttnte/linalg/tt_config.hpp"
#include "ttnte/parallel/stream_handle.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/task/stream_utils.hpp"
#include "ttnte/task/task.hpp"
#include "ttnte/utils/exception.hpp"
#include <optional>

namespace ttnte::task::cuda {

/// @brief Configure the payload for a task to transfer something to another
/// device. This calls `*destination = source.to(target_device)`.
/// @param task The task to configure with this work.
/// @param destination The place to store the resulting data.
/// @param source The data to send to the other device.
/// @param target_device The device to send the data to.
template<typename DataType>
Task& configure_transfer_task(Task& task,
  std::shared_ptr<DataType>& destination,
  const std::shared_ptr<DataType>& source, const torch::Device& target_device)
{
  if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
    // Asynchronous transfer task
    task.set_payload([destination, source, target_device,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        // Each worker thread permanently owns exactly one stream (see
        // parallel::StreamPool::claim_for_this_thread()) -- no acquire/
        // release dance, and no possibility of it being unavailable.
        active_stream = parallel::StreamPool::current_stream();
        {
          // Block stream for this send
          const auto& guard = active_stream->guard();

          // Initiate send
          *destination = source->to(target_device, true);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      TaskStatus status = test_stream(*active_stream);
      if (status == TaskStatus::COMPLETED) {
        // Reset the lambda state in case the DAG re-runs this task later.
        initiated = false;
      }
      return status;
    });

  } else if (task.get_target() == task::DeviceTarget::GPU_SYNC) {
    // Synchronous transfer task
    task.set_payload(
      [destination, source, target_device]() mutable -> TaskStatus {
        const auto& stream = parallel::StreamPool::current_stream();
        {
          // Block stream for this send
          const auto& guard = stream.guard();

          // Initiate send
          *destination = source->to(target_device, false);
          c10::impl::getDeviceGuardImpl(target_device.type())
            ->synchronizeStream(stream.stream);
        }
        return TaskStatus::COMPLETED;
      });

  } else {
    throw utils::runtime_error("ttnte::task::cuda::configure_transfer_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

/// @brief Configure the payload for a task to transfer (in-place) something to
/// another device. This calls `data.to_(target_device)`.
/// @param task The task to configure with this work.
/// @param data The data to send to the other device.
/// @param target_device The device to send the data to.
template<typename DataType>
Task& configure_transfer_task(Task& task, const std::shared_ptr<DataType>& data,
  const torch::Device& target_device)
{
  if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
    // Asynchronous transfer task
    task.set_payload([data, target_device,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        active_stream = parallel::StreamPool::current_stream();
        {
          // Block stream for this send
          const auto& guard = active_stream->guard();

          // Initiate send
          data->to_(target_device, true);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      TaskStatus status = test_stream(*active_stream);
      if (status == TaskStatus::COMPLETED) {
        initiated = false;
      }
      return status;
    });

  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    task.set_payload([data, target_device]() mutable -> TaskStatus {
      const auto& stream = parallel::StreamPool::current_stream();
      {
        // Block stream for this send
        const auto& guard = stream.guard();

        // Initiate send
        data->to_(target_device, false);
        c10::impl::getDeviceGuardImpl(target_device.type())
          ->synchronizeStream(stream.stream);
      }
      return TaskStatus::COMPLETED;
    });

  } else {
    throw utils::runtime_error("ttnte::task::cuda::configure_transfer_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

/// @brief Configure a task to execute `solver->solve(system)` on GPU.
/// @param task The task to add the payload to.
/// @param system The system passed to the `solver`.
/// @param solver The solver with a `solve()` method.
template<typename DataType, typename SolverType>
Task& configure_solve_task(Task& task, const std::shared_ptr<DataType>& system,
  const std::shared_ptr<SolverType>& solver)
{
  if (task.get_target() == DeviceTarget::GPU_ASYNC) {
    // Asynchronous solve task
    task.set_payload(
      [system, solver, active_stream = std::optional<parallel::StreamHandle>(),
        initiated = false]() mutable -> TaskStatus {
        if (!initiated) {
          active_stream = parallel::StreamPool::current_stream();
          {
            // Force all subsequent LibTorch ops onto this specific stream
            const auto& guard = active_stream->guard();

            record_stream_state(system->get_state(), active_stream->stream);

            // Dispatch solve kernel
            solver->solve(system);
          }
          initiated = true;
          return TaskStatus::POLLING;
        }

        TaskStatus status = test_stream(*active_stream);
        if (status == TaskStatus::COMPLETED) {
          initiated = false;
        }
        return status;
      });
  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    // Synchronous compute task
    task.set_payload([system, solver]() mutable -> TaskStatus {
      const auto& stream = parallel::StreamPool::current_stream();
      {
        // Force all subsequent LibTorch ops onto this specific stream
        const auto& guard = stream.guard();

        record_stream_state(system->get_state(), stream.stream);

        // Dispatch the math kernels
        solver->solve(system);
        c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
          ->synchronizeStream(stream.stream);
      }
      return TaskStatus::COMPLETED;
    });
  } else {
    throw utils::runtime_error("ttnte::task::cuda::configure_solve_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

/// @brief Configure the solution of an Eigenvalue system.
/// @param task The task to configure.
/// @param system The linear system to solve.
/// @param src The eigen source of the system.
/// @param solver The solver to use to solve the linear system.
/// @param config The low-rank tensor network rounding settings.
/// @return The configured task.
template<typename DataType, typename SourceType, typename SolverType>
Task& configure_eigensolve_task(Task& task,
  const std::shared_ptr<DataType>& system,
  const std::shared_ptr<SourceType>& src,
  const std::shared_ptr<SolverType>& solver,
  const std::shared_ptr<linalg::TTConfig>& config)
{
  if (task.get_target() == DeviceTarget::GPU_ASYNC) {
    // Asynchronous solve task
    task.set_payload([system, src, solver, config,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        active_stream = parallel::StreamPool::current_stream();
        {
          // Force all subsequent LibTorch ops onto this specific stream
          const auto& guard = active_stream->guard();

          // Compute the updated eigensource
          src->scale();

          // Dispatch solve kernel
          solver->solve(system);

          // Clear the old eigen source
          src->update(system->get_state(), config->eps, config->max_rank);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      TaskStatus status = test_stream(*active_stream);
      if (status == TaskStatus::COMPLETED) {
        initiated = false;
      }
      return status;
    });
  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    // Synchronous compute task
    task.set_payload([system, src, solver, config]() mutable -> TaskStatus {
      const auto& stream = parallel::StreamPool::current_stream();
      {
        // Force all subsequent LibTorch ops onto this specific stream
        const auto& guard = stream.guard();

        // Compute the updated eigensource
        src->scale();

        // Dispatch solve kernel
        solver->solve(system);

        // Clear the old eigen source
        src->update(system->get_state(), config->eps, config->max_rank);
        c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
          ->synchronizeStream(stream.stream);
      }
      return TaskStatus::COMPLETED;
    });
  } else {
    throw utils::runtime_error("ttnte::task::cuda::configure_solve_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

/// @brief Configure the apply task for applying a boundary operator to a
/// received boundary buffer.
/// @param task The task to configure.
/// @param coupling The boundary coupling.
/// @param config The low-rank tensor network rounding settings.
/// @return The configured task.
inline Task& configure_apply_task(Task& task,
  linalg::NeighborCoupling* coupling,
  const std::shared_ptr<linalg::TTConfig>& config)
{
  if (task.get_target() == DeviceTarget::GPU_ASYNC) {
    // Asynchronous solve task
    task.set_payload([coupling, config,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!coupling->recv_buffer.defined()) {
        return TaskStatus::COMPLETED;
      }

      if (!initiated) {
        active_stream = parallel::StreamPool::current_stream();
        {
          // Force all subsequent LibTorch ops onto this specific stream
          const auto& guard = active_stream->guard();

          record_stream_state(coupling->recv_buffer, active_stream->stream);

          // Make sure to apply the map
          coupling->set_recv_buffer(std::move(coupling->recv_buffer), true,
            config->eps, config->max_rank);

          // Apply boundary operator for this coupling
          coupling->recv_buffer =
            linalg::mv(coupling->boundary_op, coupling->recv_buffer);
          coupling->recv_buffer.round_(config->eps, config->max_rank);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      TaskStatus status = test_stream(*active_stream);
      if (status == TaskStatus::COMPLETED) {
        initiated = false;
      }
      return status;
    });
  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    // Synchronous compute task
    task.set_payload([coupling, config]() mutable -> TaskStatus {
      if (!coupling->recv_buffer.defined()) {
        return TaskStatus::COMPLETED;
      }

      const auto& stream = parallel::StreamPool::current_stream();
      {
        // Force all subsequent LibTorch ops onto this specific stream
        const auto& guard = stream.guard();

        record_stream_state(coupling->recv_buffer, stream.stream);

        // Make sure to apply the map
        coupling->set_recv_buffer(std::move(coupling->recv_buffer), true,
          config->eps, config->max_rank);

        // Apply boundary operator for this coupling
        coupling->recv_buffer =
          linalg::mv(coupling->boundary_op, coupling->recv_buffer);
        coupling->recv_buffer.round_(config->eps, config->max_rank);

        c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
          ->synchronizeStream(stream.stream);
      }
      return TaskStatus::COMPLETED;
    });
  } else {
    throw utils::runtime_error("ttnte::task::cuda::configure_apply_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

/// @brief Configure a task to extract the boundary data from the state vector
/// of a local linear system.
/// @param task The task to configure.
/// @param system The linear system to solve.
/// @param The boundary coupling.
/// @param config The low-rank tensor network rounding settings.
/// @return The configured task.
template<typename DataType>
inline Task& configure_narrow_task(Task& task,
  const std::shared_ptr<DataType>& system, linalg::NeighborCoupling* coupling,
  const std::shared_ptr<linalg::TTConfig>& config)
{
  if (task.get_target() == DeviceTarget::GPU_ASYNC) {
    // Asynchronous solve task
    task.set_payload([system, coupling, config,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        active_stream = parallel::StreamPool::current_stream();
        {
          // Force all subsequent LibTorch ops onto this specific stream
          const auto& guard = active_stream->guard();

          // Use narrow() (non-mutating) to extract the boundary face. A copy
          // of State is a shallow shared-pointer copy; narrow_() on that copy
          // would mutate the system's live state, corrupting future iterations.
          const auto& state = system->get_state();
          record_stream_state(state, active_stream->stream);
          const size_t boundary_dim = static_cast<size_t>(state.ndimension()) -
                                      coupling->connection.mapping.flip.size() -
                                      2 + coupling->dim;
          auto face =
            state.narrow(boundary_dim, coupling->is_upper ? -1 : 0, 1);
          face.round_(config->eps, config->max_rank);
          coupling->set_send_buffer(std::move(face));
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      TaskStatus status = test_stream(*active_stream);
      if (status == TaskStatus::COMPLETED) {
        initiated = false;
      }
      return status;
    });
  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    // Synchronous compute task
    task.set_payload([system, coupling, config]() mutable -> TaskStatus {
      const auto& stream = parallel::StreamPool::current_stream();
      {
        // Force all subsequent LibTorch ops onto this specific stream
        const auto& guard = stream.guard();

        // Use narrow() (non-mutating) — same reasoning as GPU_ASYNC path.
        const auto& state = system->get_state();
        record_stream_state(state, stream.stream);
        const size_t boundary_dim = static_cast<size_t>(state.ndimension()) -
                                    coupling->connection.mapping.flip.size() -
                                    2 + coupling->dim;
        auto face = state.narrow(boundary_dim, coupling->is_upper ? -1 : 0, 1);
        face.round_(config->eps, config->max_rank);
        coupling->set_send_buffer(std::move(face));
        c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
          ->synchronizeStream(stream.stream);
      }
      return TaskStatus::COMPLETED;
    });
  } else {
    throw utils::runtime_error("ttnte::task::cuda::configure_narrow_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

/// @brief Configure an arbitrary compute task to execute on GPU.
/// @param task The task to fill the payload function.
/// @param compute_kernel Function to run.
template<typename FuncType>
Task& configure_compute_task(Task& task, FuncType&& compute_kernel)
{
  if (task.get_target() == DeviceTarget::GPU_ASYNC) {
    // Asynchronous compute task
    task.set_payload([compute_kernel = std::forward<FuncType>(compute_kernel),
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        active_stream = parallel::StreamPool::current_stream();
        {
          // Force all subsequent LibTorch ops onto this specific stream
          const auto& guard = active_stream->guard();

          // Dispatch the math kernels
          compute_kernel();
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      TaskStatus status = test_stream(*active_stream);
      if (status == TaskStatus::COMPLETED) {
        initiated = false;
      }
      return status;
    });

  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    // Synchronous compute task
    task.set_payload([compute_kernel = std::forward<FuncType>(
                        compute_kernel)]() mutable -> TaskStatus {
      const auto& stream = parallel::StreamPool::current_stream();
      {
        // Force all subsequent LibTorch ops onto this specific stream
        const auto& guard = stream.guard();

        // Dispatch the math kernels
        compute_kernel();
        c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
          ->synchronizeStream(stream.stream);
      }
      return TaskStatus::COMPLETED;
    });
  } else {
    throw utils::runtime_error("ttnte::task::cuda::configure_compute_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

/// @brief Configure a buffer task send to GPU.
/// @param task The task to fill the payload function.
/// @param data A pointer to the data with a `transfer_buffer()` method.
/// @param target_device The target device is GPU however this is for
/// synchronous or asynchronous sends.
template<typename DataType>
Task& configure_transfer_buffer_task(Task& task,
  const std::shared_ptr<DataType>& data, const torch::Device& target_device)
{
  if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
    // Asynchronous transfer task
    task.set_payload([data, target_device,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        active_stream = parallel::StreamPool::current_stream();
        {
          // Block stream for this send
          const auto& guard = active_stream->guard();

          // Initiate send
          data->transfer_buffer(target_device, true);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      TaskStatus status = test_stream(*active_stream);
      if (status == TaskStatus::COMPLETED) {
        initiated = false;
      }
      return status;
    });

  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    task.set_payload([data, target_device]() mutable -> TaskStatus {
      const auto& stream = parallel::StreamPool::current_stream();
      {
        // Block stream for this send
        const auto& guard = stream.guard();

        // Initiate send
        data->transfer_buffer(target_device, false);
        c10::impl::getDeviceGuardImpl(target_device.type())
          ->synchronizeStream(stream.stream);
      }
      return TaskStatus::COMPLETED;
    });

  } else {
    throw utils::runtime_error(
      "ttnte::task::cuda::configure_transfer_buffer_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

/// @brief Configure a non-buffer task send to GPU.
/// @param task The task to fill the payload function.
/// @param data A pointer to the data with a `transfer_nonbuffer()` method.
/// @param target_device The target device is GPU however this is for
/// synchronous or asynchronous sends.
template<typename DataType>
Task& configure_transfer_nonbuffer_task(Task& task,
  const std::shared_ptr<DataType>& data, const torch::Device& target_device)
{
  if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
    // Asynchronous transfer task
    task.set_payload([data, target_device,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        active_stream = parallel::StreamPool::current_stream();
        {
          // Block stream for this send
          const auto& guard = active_stream->guard();

          // Initiate send
          data->transfer_nonbuffer(target_device, false);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      TaskStatus status = test_stream(*active_stream);
      if (status == TaskStatus::COMPLETED) {
        initiated = false;
      }
      return status;
    });

  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    task.set_payload([data, target_device]() mutable -> TaskStatus {
      const auto& stream = parallel::StreamPool::current_stream();
      {
        // Block stream for this send
        const auto& guard = stream.guard();

        // Initiate send
        data->transfer_nonbuffer(target_device, false);
        c10::impl::getDeviceGuardImpl(target_device.type())
          ->synchronizeStream(stream.stream);
      }
      return TaskStatus::COMPLETED;
    });

  } else {
    throw utils::runtime_error(
      "ttnte::task::cuda::configure_transfer_nonbuffer_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

} // namespace ttnte::task::cuda
