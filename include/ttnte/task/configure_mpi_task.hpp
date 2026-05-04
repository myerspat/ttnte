#pragma once

#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/request.hpp"
#include "ttnte/task/task.hpp"
#include <torch/extension.h>

namespace ttnte::task::mpi {

/// @brief Method for converting torch types to the ttnte::parallel::DataType
/// type.
/// @param st The at::ScalarType data type.
/// @return The resulting ttnte::parallel::DataType data type.
inline parallel::DataType torch2ttnte(at::ScalarType st)
{
  switch (st) {
  case at::kDouble:
    return parallel::DataType::DOUBLE;
  case at::kFloat:
    return parallel::DataType::FLOAT;
  case at::kInt:
    return parallel::DataType::INT32;
  case at::kLong:
    return parallel::DataType::INT64;
  default:
    return parallel::DataType::BYTE;
  }
}

/// @brief Test if the request has completed. If it has reset the task's mutable
/// data.
/// @param request The request.
/// @param initiated Whether the request has started. We reset this when the
/// request finishes.
/// @return The status of the task.
TaskStatus inline test_request(parallel::Request& request, bool& initiated)
{
  if (request.test()) {
    // Reset the members of this lambda in case we want to call it again later
    request = parallel::Request(); // Empty/null request
    initiated = false;
    return TaskStatus::COMPLETED;
  }

  return TaskStatus::POLLING;
}

/// @brief Test if the request has completed. If it has reset the task's mutable
/// data.
/// @param request The request.
/// @param initiated Whether the request has started. We reset this when the
/// request finishes.
/// @param packed_tensor The torch tensor used by the request. We set this to an
/// undefined tensor once the task finishes.
/// @return The status of the task.
TaskStatus inline test_request(
  parallel::Request& request, bool& initiated, torch::Tensor& packed_tensor)
{
  if (request.test()) {
    // Reset the members of this lambda in case we want to call it again later
    request = parallel::Request(); // Empty/null request
    packed_tensor = torch::Tensor();
    initiated = false;
    return TaskStatus::COMPLETED;
  }

  return TaskStatus::POLLING;
}

/// @brief Configure an MPI non-blocking all-gather task.
/// @param task The task who's payload will be updated.
/// @param send_buffer The buffer to send.
/// @param send_count The length of the send buffer.
/// @param recv_buffer The buffer to receive the gathered data from other ranks.
/// @param recv_count The length of the receive buffer.
/// @param comm The communicator for access to MPI.
template<typename BufferType>
Task& configure_iallgather_task(Task& task, const BufferType* send_buffer,
  int send_count, BufferType* recv_buffer, int recv_count,
  const parallel::Communicator& comm)
{
  // Set the target to be the network for communication
  task.set_target(DeviceTarget::NETWORK_ASYNC);

  // Set the payload function
  task.set_payload([send_buffer, send_count, recv_buffer, recv_count, &comm,
                     request = parallel::Request(),
                     initiated = false]() mutable -> TaskStatus {
    if (!initiated) {
      // Send the buffer if we haven't already
      request =
        comm.iallgather(send_buffer, send_count, recv_buffer, recv_count);
      return TaskStatus::POLLING;
    }

    // Test request and return result
    return test_request(request, initiated);
  });

  return task;
}

/// @brief Configure an MPI non-blocking send task.
/// @param task The task who's payload will be updated.
/// @param send_buffer The buffer to send.
/// @param count The length of the send buffer.
/// @param target_rank The rank which recieve the buffer.
/// @param tag The tag for the data.
/// @param comm The communicator for access to MPI.
template<typename BufferType, typename TagType>
Task& configure_isend_task(Task& task, const BufferType* send_buffer, int count,
  int target_rank, TagType tag, const parallel::Communicator& comm)
{
  task.set_target(DeviceTarget::NETWORK_ASYNC);

  task.set_payload(
    [send_buffer, count, target_rank, tag, &comm, request = parallel::Request(),
      initiated = false]() mutable -> TaskStatus {
      if (!initiated) {
        request = comm.isend(send_buffer, count, target_rank, tag);
        initiated = true;
        return TaskStatus::POLLING;
      }
      return test_request(request, initiated);
    });

  return task;
}

/// @brief Configure an MPI non-blocking recieve task.
/// @param task The task who's payload will be updated.
/// @param recv_buffer The buffer to recieve the information in.
/// @param count The length of the buffer of data.
/// @param target_rank The rank which is sending the buffer of data.
/// @param tag The tag for the data.
/// @param comm The communicator for access to MPI.
template<typename BufferType, typename TagType>
Task& configure_irecv_task(Task& task, BufferType* recv_buffer, int count,
  int target_rank, TagType tag, const parallel::Communicator& comm)
{
  task.set_target(DeviceTarget::NETWORK_ASYNC);

  task.set_payload([recv_buffer, count, target_rank, tag, &comm,
                     request = ttnte::parallel::Request(),
                     initiated = false]() mutable -> TaskStatus {
    if (!initiated) {
      request = comm.irecv(recv_buffer, count, target_rank, tag);
      initiated = true;
      return TaskStatus::POLLING;
    }
    return test_request(request, initiated);
  });

  return task;
}

/// @brief Configure an MPI non-blocking send without prior knowledge of its
/// size.
/// @param task The task who's payload will be updated.
/// @param send_tensor The torch tensor to send.
/// @param target_rank The MPI rank to send the buffer of data to.
/// @param tag The tag for the data.
/// @param comm The communicator for access to MPI.
template<typename TagType>
Task& configure_dynamic_isend_task(Task& task, const torch::Tensor* send_tensor,
  int target_rank, TagType tag, const parallel::Communicator& comm)
{
  task.set_target(DeviceTarget::NETWORK_ASYNC);

  task.set_payload([send_tensor, target_rank, tag, &comm,
                     request = parallel::Request(), initiated = false,
                     packed_tensor = torch::Tensor()]() mutable -> TaskStatus {
    if (!initiated) {
      // Ensure contiguous
      packed_tensor = send_tensor->contiguous();
      int count = packed_tensor.numel();

      // DISPATCH: This macro handles the switch/case for all Libtorch types
      AT_DISPATCH_ALL_TYPES(packed_tensor.scalar_type(), "dynamic_isend", ([&] {
        request = comm.isend(
          packed_tensor.data_ptr<scalar_t>(), count, target_rank, tag);
      }));

      initiated = true;
      return TaskStatus::POLLING;
    }

    return test_request(request, initiated, packed_tensor);
  });

  return task;
}

/// @brief Configure an MPI non-blocking recieve without prior knowledge of its
/// size.
/// @param task The task who's payload will be updated.
/// @param recv_tensor The tensor to recieve the buffer of data.
/// @param source_rank The MPI rank sending the information.
/// @param tag The tag for the data.
/// @param comm The communicator for access to MPI.
template<typename TagType>
Task& configure_dynamic_irecv_task(Task& task, torch::Tensor* recv_buffer,
  int source_rank, TagType tag, const parallel::Communicator& comm)
{
  task.set_target(DeviceTarget::NETWORK_ASYNC);

  // Capture the type of the buffer
  const auto expected_type = recv_buffer->scalar_type();

  task.set_payload([recv_buffer, source_rank, tag, &comm, expected_type,
                     request = parallel::Request(),
                     probe = parallel::ProbeResult {false, 0, 0},
                     initiated = false]() mutable -> TaskStatus {
    if (!initiated) {
      // Probe: look for the message in the queue
      probe = comm.iprobe(source_rank, tag, torch2ttnte(expected_type));

      // Return if we didn't find a match
      if (!probe.matched) {
        return TaskStatus::POLLING;
      }

      AT_DISPATCH_ALL_TYPES(expected_type, "dynamic_irecv", [&] {
        // Allocate tensor at runtime
        *recv_buffer = torch::empty(
          {static_cast<int64_t>(probe.count)}, torch::dtype<scalar_t>());

        // Receive the data in the tensor memory
        request = comm.imrecv(recv_buffer->data_ptr<scalar_t>(), probe.count,
          torch2ttnte(expected_type), probe);
      });

      initiated = true;
      return TaskStatus::POLLING;
    }

    return test_request(request, initiated);
  });

  return task;
}

} // namespace ttnte::task::mpi
