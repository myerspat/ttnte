#include "ttnte/linalg/tt_state.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::linalg {

// =================================================================
// Protected methods
State::Ptr TTState::to_impl(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return TTState::create(
    tt_vector_.to(device, dtype, non_blocking, copy, memory_format), label_);
}

void TTState::check_cores() const
{
  const auto& cores = get_cores();

  for (const auto& core : cores) {
    if (core.size(2) != 1) {
      throw utils::runtime_error(*this, "ttnte::linalg::TTState::check_cores",
        "The second dimension is not size 1 for all cores");
    }
  }
}

// =================================================================
// Public methods
TTState::Ptr TTState::clone_from(
  const TTEngine::Tensors& cores, std::optional<std::string> label)
{
  return TTState::create(TTEngine::clone_from(cores, true),
    label.has_value() ? Label::from_string(label.value())
                      : Label::create_internal());
}

TTState::Ptr TTState::clone_from(
  const TTEngine::Tensors& cores, const Label& label)
{
  return TTState::create(TTEngine::clone_from(cores, true), label);
}

TTState::Ptr TTState::to(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return TTState::create(
    tt_vector_.to(device, dtype, non_blocking, copy, memory_format), label_);
}

TTState::Ptr TTState::to(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return TTState::create(
    tt_vector_.to(get_device(), dtype, non_blocking, copy, memory_format),
    label_);
}

TTState::Ptr TTState::to(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return TTState::create(
    tt_vector_.to(device, get_dtype(), non_blocking, copy, memory_format),
    label_);
}

void TTState::to_(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  tt_vector_.to_(device, dtype, non_blocking, copy, memory_format);
}

void TTState::to_(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  tt_vector_.to_(dtype, non_blocking, copy, memory_format);
};

void TTState::to_(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  tt_vector_.to_(device, non_blocking, copy, memory_format);
};

} // namespace ttnte::linalg
