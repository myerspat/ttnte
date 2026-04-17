#include "ttnte/linalg/state.hpp"

namespace ttnte::linalg {

State::~State() = default;

// =================================================================
// Public methods
State::Ptr State::to(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return to_impl(device, dtype, non_blocking, copy, memory_format);
}

State::Ptr State::to(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return to_impl(get_device(), dtype, non_blocking, copy, memory_format);
}

State::Ptr State::to(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return to_impl(device, get_dtype(), non_blocking, copy, memory_format);
}

} // namespace ttnte::linalg
