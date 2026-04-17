#include "ttnte/linalg/tt_operator.hpp"

namespace ttnte::linalg {

// =================================================================
// Protected methods
Operator::Ptr TTOperator::to_impl(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return TTOperator::create(
    tt_matrix_.to(device, dtype, non_blocking, copy, memory_format), label_);
}

// =================================================================
// Public methods
TTOperator::Ptr TTOperator::clone_from(
  const TTEngine::Tensors& cores, std::optional<std::string> label)
{
  return TTOperator::create(TTEngine::clone_from(cores, true),
    label.has_value() ? Label::from_string(label.value())
                      : Label::create_internal());
}

TTOperator::Ptr TTOperator::clone_from(
  const TTEngine::Tensors& cores, const Label& label)
{
  return TTOperator::create(TTEngine::clone_from(cores, true), label);
}

TTOperator::Ptr TTOperator::to(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return TTOperator::create(
    tt_matrix_.to(device, dtype, non_blocking, copy, memory_format), label_);
}

TTOperator::Ptr TTOperator::to(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return TTOperator::create(
    tt_matrix_.to(get_device(), dtype, non_blocking, copy, memory_format),
    label_);
}

TTOperator::Ptr TTOperator::to(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return TTOperator::create(
    tt_matrix_.to(device, get_dtype(), non_blocking, copy, memory_format),
    label_);
}

void TTOperator::to_(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  tt_matrix_.to_(device, dtype, non_blocking, copy, memory_format);
}

void TTOperator::to_(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  tt_matrix_.to_(dtype, non_blocking, copy, memory_format);
};

void TTOperator::to_(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  tt_matrix_.to_(device, non_blocking, copy, memory_format);
};

void TTOperator::from_buffer(const torch::Tensor& buffer)
{
  tt_matrix_.from_buffer(buffer);
}

} // namespace ttnte::linalg
