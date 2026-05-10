#include "ttnte/linalg/state.hpp"
#include "ttnte/linalg/tt_state.hpp"
#include "ttnte/parallel/boundary_buffer.hpp"
#include "ttnte/utils/exception.hpp"

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

State::Ptr State::unpack(const torch::Tensor& buffer, bool clone)
{
  // Get the type ID from the buffer
  auto type_id = static_cast<parallel::BoundaryType>(buffer[0].item<int64_t>());

  switch (type_id) {
  case parallel::BoundaryType::TENSOR_TRAIN:
    return TTState::unpack(buffer, clone);
  default:
    throw utils::runtime_error("ttnte::linalg::State::unpack",
      "State type with ID " + std::to_string(static_cast<int8_t>(type_id)) +
        " is not supported");
  }
}

} // namespace ttnte::linalg
