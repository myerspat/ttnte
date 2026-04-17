#include "ttnte/linalg/operator.hpp"

namespace ttnte::linalg {

Operator::~Operator() = default;

// =================================================================
// Public methods
Operator::Ptr Operator::to(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return to_impl(device, dtype, non_blocking, copy, memory_format);
}

Operator::Ptr Operator::to(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return to_impl(get_device(), dtype, non_blocking, copy, memory_format);
}

Operator::Ptr Operator::to(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return to_impl(device, get_dtype(), non_blocking, copy, memory_format);
}

} // namespace ttnte::linalg
