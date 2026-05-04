#pragma once

#include "ttnte/parallel/stream_handle.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/task/task.hpp"
#include "ttnte/utils/exception.hpp"
#include <optional>

namespace {

/// @brief Test an active stream to see if it has completed.
/// @param device_type The device currently running the task.
/// @param stream_pool The collection of available streams. When `active_stream`
/// completes we give it back to the stream pool.
/// @param active_stream The working stream to the GPU.
/// @param initiated Whether the task has started or not. This is reset to false
/// once the `active_stream` completes.
inline ttnte::task::TaskStatus test_stream(const c10::DeviceType& device_type,
  const ttnte::parallel::StreamPool::Ptr& stream_pool,
  std::optional<ttnte::parallel::StreamHandle>& active_stream, bool& initiated)
{
  assert(active_stream.has_value());

  // Ask libtorch how the stream is doing (will throw an error for us)
  bool is_done = c10::impl::getDeviceGuardImpl(device_type)
                   ->queryStream(active_stream->stream);

  if (is_done) {
    // Reset the lambda state in case the DAG re-runs this task later.
    initiated = false;
    stream_pool->release(*active_stream);
    active_stream = std::nullopt;
    return ttnte::task::TaskStatus::COMPLETED;

  } else {
    // The GPU is still working / data is still moving.
    return ttnte::task::TaskStatus::POLLING;
  }
}

} // namespace

namespace ttnte::task::cuda {

/// @brief Configure the payload for a task to transfer something to another
/// device. This calls `*destination = source.to(target_device)`.
/// @param task The task to configure with this work.
/// @param destination The place to store the resulting data.
/// @param source The data to send to the other device.
/// @param target_device The device to send the data to.
/// @param stream_pool The stream pool to pull an active stream from.
template<typename DataType>
Task& configure_transfer_task(Task& task,
  std::shared_ptr<DataType>& destination,
  const std::shared_ptr<DataType>& source, const torch::Device& target_device,
  const parallel::StreamPool::Ptr& stream_pool)
{
  if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
    // Asynchronous transfer task
    task.set_payload([destination, source, target_device, stream_pool,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        // Get an available stream and return early if there isn't one available
        active_stream = stream_pool->try_acquire();
        if (!active_stream.has_value()) {
          return TaskStatus::POLLING;
        }
        {
          // Block stream for this send
          const auto& guard = active_stream->guard();

          // Initiate send
          *destination = source->to(target_device, true);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      return test_stream(
        target_device.type(), stream_pool, active_stream, initiated);
    });

  } else if (task.get_target() == task::DeviceTarget::GPU_SYNC) {
    // Synchronous transfer task
    task.set_payload([destination, source, target_device,
                       stream_pool]() mutable -> TaskStatus {
      // Get an available stream
      auto stream = stream_pool->try_acquire();
      if (!stream.has_value()) {
        return TaskStatus::POLLING;
      }
      {
        // Block stream for this send
        const auto& guard = stream->guard();

        // Initiate send
        *destination = source->to(target_device, false);
        c10::impl::getDeviceGuardImpl(target_device.type())
          ->synchronizeStream(stream->stream);
      }
      stream_pool->release(*stream);
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
/// @param stream_pool The stream pool to pull an active stream from.
template<typename DataType>
Task& configure_transfer_task(Task& task, const std::shared_ptr<DataType>& data,
  const torch::Device& target_device,
  const parallel::StreamPool::Ptr& stream_pool)
{
  if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
    // Asynchronous transfer task
    task.set_payload([data, target_device, stream_pool,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        // Get an available stream and return early if there isn't one available
        active_stream = stream_pool->try_acquire();
        if (!active_stream.has_value()) {
          return TaskStatus::POLLING;
        }
        {
          // Block stream for this send
          const auto& guard = active_stream->guard();

          // Initiate send
          data->to_(target_device, true);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      return test_stream(
        target_device.type(), stream_pool, active_stream, initiated);
    });

  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    task.set_payload(
      [data, target_device, stream_pool]() mutable -> TaskStatus {
        // Get an available stream
        auto stream = stream_pool->try_acquire();
        if (!stream.has_value()) {
          return TaskStatus::POLLING;
        }
        {
          // Block stream for this send
          const auto& guard = stream->guard();

          // Initiate send
          data->to_(target_device, false);
          c10::impl::getDeviceGuardImpl(target_device.type())
            ->synchronizeStream(stream->stream);
        }

        stream_pool->release(*stream);
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
/// @param stream_pool The stream pool to pull an active stream from.
template<typename DataType, typename SolverType>
Task& configure_solve_task(Task& task, const std::shared_ptr<DataType>& system,
  const std::shared_ptr<SolverType>& solver,
  const parallel::StreamPool::Ptr& stream_pool)
{
  if (task.get_target() == DeviceTarget::GPU_ASYNC) {
    // Asynchronous solve task
    task.set_payload([system, solver, stream_pool,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        // Get an available stream and return early if there isn't one
        // available
        active_stream = stream_pool->try_acquire();
        if (!active_stream.has_value()) {
          return TaskStatus::POLLING;
        }
        {
          // Force all subsequent LibTorch ops onto this specific stream
          const auto& guard = active_stream->guard();

          // Dispatch solve kernel
          solver->solve(system);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      return test_stream(
        c10::DeviceType::CUDA, stream_pool, active_stream, initiated);
    });
  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    // Synchronous compute task
    task.set_payload([system, solver, stream_pool]() mutable -> TaskStatus {
      // Get an available stream
      auto stream = stream_pool->try_acquire();
      if (!stream.has_value()) {
        return TaskStatus::POLLING;
      }
      {
        // Force all subsequent LibTorch ops onto this specific stream
        const auto& guard = stream->guard();

        // Dispatch the math kernels
        solver->solve(system);
        c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
          ->synchronizeStream(stream->stream);
      }
      stream_pool->release(*stream);
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

/// @brief Configure an arbitrary compute task to execute on GPU.
/// @param task The task to fill the payload function.
/// @param compute_kernel Function to run.
/// @param stream_pool Pool of GPU streams to pull an available one for
/// execution.
template<typename FuncType>
Task& configure_compute_task(Task& task, FuncType&& compute_kernel,
  const parallel::StreamPool::Ptr stream_pool)
{
  if (task.get_target() == DeviceTarget::GPU_ASYNC) {
    // Asynchronous compute task
    task.set_payload(
      [compute_kernel = std::forward<FuncType>(compute_kernel), stream_pool,
        active_stream = std::optional<parallel::StreamHandle>(),
        initiated = false]() mutable -> TaskStatus {
        if (!initiated) {
          // Get an available stream and return early if there isn't one
          // available
          active_stream = stream_pool->try_acquire();
          if (!active_stream.has_value()) {
            return TaskStatus::POLLING;
          }
          {
            // Force all subsequent LibTorch ops onto this specific stream
            const auto& guard = active_stream->guard();

            // Dispatch the math kernels
            compute_kernel();
          }
          initiated = true;
          return TaskStatus::POLLING;
        }

        return test_stream(
          c10::DeviceType::CUDA, stream_pool, active_stream, initiated);
      });

  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    // Synchronous compute task
    task.set_payload([compute_kernel = std::forward<FuncType>(compute_kernel),
                       stream_pool]() mutable -> TaskStatus {
      // Get an available stream
      auto stream = stream_pool->try_acquire();
      if (!stream.has_value()) {
        return TaskStatus::POLLING;
      }
      {
        // Force all subsequent LibTorch ops onto this specific stream
        const auto& guard = stream->guard();

        // Dispatch the math kernels
        compute_kernel();
        c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
          ->synchronizeStream(stream->stream);
      }
      stream_pool->release(*stream);
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
/// @param stream_pool A pointer to the GPU stream pool manager.
template<typename DataType>
Task& configure_transfer_buffer_task(Task& task,
  const std::shared_ptr<DataType>& data, const torch::Device& target_device,
  const parallel::StreamPool::Ptr& stream_pool)
{
  if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
    // Asynchronous transfer task
    task.set_payload([data, target_device, stream_pool,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        // Get an available stream and return early if there isn't one
        // available
        active_stream = stream_pool->try_acquire();
        if (!active_stream.has_value()) {
          return TaskStatus::POLLING;
        }
        {
          // Block stream for this send
          const auto& guard = active_stream->guard();

          // Initiate send
          data->transfer_buffer(target_device, true);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      return test_stream(
        target_device.type(), stream_pool, active_stream, initiated);
    });

  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    task.set_payload(
      [data, target_device, stream_pool]() mutable -> TaskStatus {
        // Get an available stream
        auto stream = stream_pool->try_acquire();
        if (!stream.has_value()) {
          return TaskStatus::POLLING;
        }
        {
          // Block stream for this send
          const auto& guard = stream->guard();

          // Initiate send
          data->transfer_buffer(target_device, false);
          c10::impl::getDeviceGuardImpl(target_device.type())
            ->synchronizeStream(stream->stream);
        }

        stream_pool->release(*stream);
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
/// @param stream_pool A pointer to the GPU stream pool manager.
template<typename DataType>
Task& configure_transfer_nonbuffer_task(Task& task,
  const std::shared_ptr<DataType>& data, const torch::Device& target_device,
  const parallel::StreamPool::Ptr& stream_pool)
{
  if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
    // Asynchronous transfer task
    task.set_payload([data, target_device, stream_pool,
                       active_stream = std::optional<parallel::StreamHandle>(),
                       initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        // Get an available stream and return early if there isn't one
        // available
        active_stream = stream_pool->try_acquire();
        if (!active_stream.has_value()) {
          return TaskStatus::POLLING;
        }
        {
          // Block stream for this send
          const auto& guard = active_stream->guard();

          // Initiate send
          data->transfer_nonbuffer(target_device, true);
        }
        initiated = true;
        return TaskStatus::POLLING;
      }

      return test_stream(
        target_device.type(), stream_pool, active_stream, initiated);
    });

  } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
    task.set_payload(
      [data, target_device, stream_pool]() mutable -> TaskStatus {
        // Get an available stream
        auto stream = stream_pool->try_acquire();
        if (!stream.has_value()) {
          return TaskStatus::POLLING;
        }
        {
          // Block stream for this send
          const auto& guard = stream->guard();

          // Initiate send
          data->transfer_nonbuffer(target_device, false);
          c10::impl::getDeviceGuardImpl(target_device.type())
            ->synchronizeStream(stream->stream);
        }

        stream_pool->release(*stream);
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
