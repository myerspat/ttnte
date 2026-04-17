#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/utils/exception.hpp"
#include <c10/core/DefaultDtype.h>

namespace ttnte::linalg {

// =================================================================
// Protected methods
void LinearSystem::is_finalized_or_error(const std::string& func_name) const
{
  if (!is_finalized_) {
    throw utils::runtime_error(*this, error_context(func_name),
      "The linear system has not been finalized yet");
  }
}

// =================================================================
// Public methods
// LinearSystem::Ptr LinearSystem::to(const torch::Device& device,
//   const at::ScalarType& dtype, bool non_blocking, bool copy,
//   std::optional<at::MemoryFormat> memory_format)
// {
//   return to_impl(device, dtype, non_blocking, copy, memory_format);
// }
//
// LinearSystem::Ptr LinearSystem::to(const at::ScalarType& dtype,
//   bool non_blocking, bool copy, std::optional<at::MemoryFormat>
//   memory_format)
// {
//   return to_impl(dtype, non_blocking, copy, memory_format);
// }
//
// LinearSystem::Ptr LinearSystem::to(const torch::Device& device,
//   bool non_blocking, bool copy, std::optional<at::MemoryFormat>
//   memory_format)
// {
//   return to_impl(device, get_dtype(), non_blocking, copy, memory_format);
// }

void LinearSystem::transfer_buffer(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  is_finalized_or_error("transfer_buffer");

  if (device.is_cuda()) {
    // If the device buffer is empty then we send what's in the host buffer
    if (!device_buffer_.defined()) {
      device_buffer_ =
        host_buffer_.to(device, dtype, non_blocking, copy, memory_format);
    } else {
      device_buffer_ =
        device_buffer_.to(device, dtype, non_blocking, copy, memory_format);
    }

  } else {
    if (!host_buffer_.defined()) {
      // Send data from device to host and then clear the device
      host_buffer_ =
        device_buffer_.to(device, dtype, non_blocking, copy, memory_format)
          .pin_memory();
    } else {
      host_buffer_ =
        host_buffer_.to(device, dtype, non_blocking, copy, memory_format);
    }

    // Clear the device data
    device_buffer_ = torch::Tensor();
  }
}

void LinearSystem::transfer_buffer(const torch::Device& device,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  is_finalized_or_error("transfer_buffer");
  transfer_buffer(device,
    host_buffer_.defined() ? host_buffer_.scalar_type()
                           : device_buffer_.scalar_type(),
    non_blocking, copy, memory_format);
}

} // namespace ttnte::linalg
