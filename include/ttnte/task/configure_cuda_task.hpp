#pragma once

#include "ttnte/parallel/stream_handle.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/task/task.hpp"
#include "ttnte/utils/exception.hpp"
#include <optional>

namespace {

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

// inline TaskStatus test_stream(const parallel::CudaStreamPool::Ptr&
// stream_pool,
//   std::optional<c10::cuda::CUDAStream>& active_stream, bool& initiated)
// {
//   assert(active_stream.has_value());
//   cudaError_t status = cudaStreamQuery(active_stream->stream());
//
//   if (status == cudaSuccess) {
//     // Reset the lambda state in case the DAG re-runs this task later.
//     initiated = false;
//     stream_pool->release(*active_stream);
//     active_stream = std::nullopt;
//     return TaskStatus::COMPLETED;
//
//   } else if (status == cudaErrorNotReady) {
//     // The GPU is still working / data is still moving.
//     return TaskStatus::POLLING;
//
//   } else {
//     // A catastrophic hardware or kernel failure occurred.
//     throw utils::runtime_error("ttnte::task::cuda::test_stream",
//       "CUDA error during stream polling:\n" + std::to_string(status));
//   }
// }

// template<typename DataType>
// Task& configure_transfer_task(Task& task,
//   std::shared_ptr<DataType>& destination, const DataType& source,
//   const torch::Device& target_device,
//   const parallel::StreamPool::Ptr& stream_pool)
// {
//   if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
//     // Asynchronous transfer task
//     task.set_payload([destination, source, target_device, stream_pool,
//                        active_stream =
//                        std::optional<c10::cuda::CUDAStream>(), initiated =
//                        false]() mutable -> TaskStatus {
//       if (!initiated) {
//         // Get an available stream and return early if there isn't one
//         available active_stream = stream_pool->try_acquire(); if
//         (!active_stream.has_value()) {
//           return TaskStatus::POLLING;
//         }
//
//         {
//           // Block stream for this send
//           c10::cuda::CUDAStreamGuard guard(*active_stream);
//
//           // Initiate send
//           destination = source->to(target_device, true);
//         }
//         initiated = true;
//         return TaskStatus::POLLING;
//       }
//
//       return test_stream(stream_pool, active_stream, initiated);
//     });
//
//   } else if (task.get_target() == task::DeviceTarget::GPU_SYNC) {
//     // Synchronous transfer task
//     task.set_payload([destination, source, target_device,
//                        stream_pool]() mutable -> TaskStatus {
//       // Get an available stream
//       auto stream = stream_pool->try_acquire();
//       if (!stream.has_value()) {
//         return TaskStatus::POLLING;
//       }
//       {
//         // Block stream for this send
//         c10::cuda::CUDAStreamGuard guard(*stream);
//
//         // Initiate send
//         destination = source->to(target_device, false);
//         cudaStreamSynchronize(stream->stream());
//       }
//       stream_pool->release(*stream);
//       return TaskStatus::COMPLETED;
//     });
//
//   } else {
//     throw
//     utils::runtime_error("ttnte::task::cuda::configure_transfer_task\n",
//       "The target device of the `task` must be either\n"
//       "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
//       "`ttnte::task::DeviceTarget::GPU_ASYNC`");
//   }
//
//   return task;
// }

// template<typename DataType>
// Task& configure_transfer_task(Task& task, const std::shared_ptr<DataType>&
// data,
//   const torch::Device& target_device,
//   const parallel::CudaStreamPool::Ptr& stream_pool)
// {
//   if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
//     // Asynchronous transfer task
//     task.set_payload([data, target_device, stream_pool,
//                        active_stream =
//                        std::optional<c10::cuda::CUDAStream>(), initiated =
//                        false]() mutable -> TaskStatus {
//       if (!initiated) {
//         // Get an available stream and return early if there isn't one
//         available active_stream = stream_pool->try_acquire(); if
//         (!active_stream.has_value()) {
//           return TaskStatus::POLLING;
//         }
//         {
//           // Block stream for this send
//           c10::cuda::CUDAStreamGuard guard(*active_stream);
//
//           // Initiate send
//           data->to_(target_device, true);
//         }
//         initiated = true;
//         return TaskStatus::POLLING;
//       }
//
//       return test_stream(stream_pool, active_stream, initiated);
//     });
//
//   } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
//     task.set_payload(
//       [data, target_device, stream_pool]() mutable -> TaskStatus {
//         // Get an available stream
//         auto stream = stream_pool->try_acquire();
//         if (!stream.has_value()) {
//           return TaskStatus::POLLING;
//         }
//         {
//           // Block stream for this send
//           c10::cuda::CUDAStreamGuard guard(*stream);
//
//           // Initiate send
//           data->to_(target_device, false);
//           cudaStreamSynchronize(stream->stream());
//         }
//
//         stream_pool->release(*stream);
//         return TaskStatus::COMPLETED;
//       });
//
//   } else {
//     throw
//     utils::runtime_error("ttnte::task::cuda::configure_transfer_task\n",
//       "The target device of the `task` must be either\n"
//       "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
//       "`ttnte::task::DeviceTarget::GPU_ASYNC`");
//   }
//
//   return task;
// }

// template<typename FuncType>
// Task& configure_compute_task(Task& task, FuncType&& compute_kernel,
//   const parallel::CudaStreamPool::Ptr stream_pool)
// {
//   if (task.get_target() == DeviceTarget::GPU_ASYNC) {
//     // Asynchronous compute task
//     task.set_payload(
//       [compute_kernel = std::forward<FuncType>(compute_kernel), stream_pool,
//         active_stream = std::optional<c10::cuda::CUDAStream>(),
//         initiated = false]() mutable -> TaskStatus {
//         if (!initiated) {
//           // Get an available stream and return early if there isn't one
//           // available
//           active_stream = stream_pool->try_acquire();
//           if (!active_stream.has_value()) {
//             return TaskStatus::POLLING;
//           }
//           {
//             // Force all subsequent LibTorch ops onto this specific stream
//             c10::cuda::CUDAStreamGuard guard(*active_stream);
//
//             // Dispatch the math kernels
//             compute_kernel();
//           }
//           initiated = true;
//           return TaskStatus::POLLING;
//         }
//
//         return test_stream(stream_pool, active_stream, initiated);
//       });
//
//   } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
//     // Synchronous compute task
//     task.set_payload([compute_kernel =
//     std::forward<FuncType>(compute_kernel),
//                        stream_pool]() mutable -> TaskStatus {
//       // Get an available stream
//       auto stream = stream_pool->try_acquire();
//       if (!stream.has_value()) {
//         return TaskStatus::POLLING;
//       }
//       {
//         // Force all subsequent LibTorch ops onto this specific stream
//         c10::cuda::CUDAStreamGuard guard(*stream);
//
//         // Dispatch the math kernels
//         compute_kernel();
//         cudaStreamSynchronize(stream->stream());
//       }
//       stream_pool->release(*stream);
//       return TaskStatus::COMPLETED;
//     });
//   } else {
//     throw utils::runtime_error("ttnte::task::cuda::configure_compute_task\n",
//       "The target device of the `task` must be either\n"
//       "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
//       "`ttnte::task::DeviceTarget::GPU_ASYNC`");
//   }
//
//   return task;
// }

// template<typename DataType>
// Task& configure_transfer_buffer_task(Task& task,
//   const std::shared_ptr<DataType>& data, const torch::Device& target_device,
//   const parallel::CudaStreamPool::Ptr& stream_pool)
// {
//   if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
//     // Asynchronous transfer task
//     task.set_payload([data, target_device, stream_pool,
//                        active_stream =
//                        std::optional<c10::cuda::CUDAStream>(), initiated =
//                        false]() mutable -> TaskStatus {
//       if (!initiated) {
//         // Get an available stream and return early if there isn't one
//         // available
//         active_stream = stream_pool->try_acquire();
//         if (!active_stream.has_value()) {
//           return TaskStatus::POLLING;
//         }
//         {
//           // Block stream for this send
//           c10::cuda::CUDAStreamGuard guard(*active_stream);
//
//           // Initiate send
//           data->transfer_buffer(target_device, true);
//         }
//         initiated = true;
//         return TaskStatus::POLLING;
//       }
//
//       return test_stream(stream_pool, active_stream, initiated);
//     });
//
//   } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
//     task.set_payload(
//       [data, target_device, stream_pool]() mutable -> TaskStatus {
//         // Get an available stream
//         auto stream = stream_pool->try_acquire();
//         if (!stream.has_value()) {
//           return TaskStatus::POLLING;
//         }
//         {
//           // Block stream for this send
//           c10::cuda::CUDAStreamGuard guard(*stream);
//
//           // Initiate send
//           data->transfer_buffer(target_device, false);
//           cudaStreamSynchronize(stream->stream());
//         }
//
//         stream_pool->release(*stream);
//         return TaskStatus::COMPLETED;
//       });
//
//   } else {
//     throw utils::runtime_error(
//       "ttnte::task::cuda::configure_transfer_buffer_task\n",
//       "The target device of the `task` must be either\n"
//       "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
//       "`ttnte::task::DeviceTarget::GPU_ASYNC`");
//   }
//
//   return task;
// }
//
// template<typename DataType>
// Task& configure_transfer_nonbuffer_task(Task& task,
//   const std::shared_ptr<DataType>& data, const torch::Device& target_device,
//   const parallel::CudaStreamPool::Ptr& stream_pool)
// {
//   if (task.get_target() == task::DeviceTarget::GPU_ASYNC) {
//     // Asynchronous transfer task
//     task.set_payload([data, target_device, stream_pool,
//                        active_stream =
//                        std::optional<c10::cuda::CUDAStream>(), initiated =
//                        false]() mutable -> TaskStatus {
//       if (!initiated) {
//         // Get an available stream and return early if there isn't one
//         // available
//         active_stream = stream_pool->try_acquire();
//         if (!active_stream.has_value()) {
//           return TaskStatus::POLLING;
//         }
//         {
//           // Block stream for this send
//           c10::cuda::CUDAStreamGuard guard(*active_stream);
//
//           // Initiate send
//           data->transfer_nonbuffer(target_device, true);
//         }
//         initiated = true;
//         return TaskStatus::POLLING;
//       }
//
//       return test_stream(stream_pool, active_stream, initiated);
//     });
//
//   } else if (task.get_target() == DeviceTarget::GPU_SYNC) {
//     task.set_payload(
//       [data, target_device, stream_pool]() mutable -> TaskStatus {
//         // Get an available stream
//         auto stream = stream_pool->try_acquire();
//         if (!stream.has_value()) {
//           return TaskStatus::POLLING;
//         }
//         {
//           // Block stream for this send
//           c10::cuda::CUDAStreamGuard guard(*stream);
//
//           // Initiate send
//           data->transfer_nonbuffer(target_device, false);
//           cudaStreamSynchronize(stream->stream());
//         }
//
//         stream_pool->release(*stream);
//         return TaskStatus::COMPLETED;
//       });
//
//   } else {
//     throw utils::runtime_error(
//       "ttnte::task::cuda::configure_transfer_nonbuffer_task\n",
//       "The target device of the `task` must be either\n"
//       "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
//       "`ttnte::task::DeviceTarget::GPU_ASYNC`");
//   }
//
//   return task;
// }
