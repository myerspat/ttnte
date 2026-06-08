#include "ttnte/linalg/state.hpp"
#include "ttnte/utils/exception.hpp"
#include <cstring>

namespace ttnte::linalg {

// =================================================================
// Public methods

void State::to_buffer(const torch::Tensor& buffer) const
{
  std::visit([&](const auto& v) { v.to_buffer(buffer); }, get_variant());
}

void State::from_buffer(const torch::Tensor& buffer)
{
  std::visit([&](auto& v) { v.from_buffer(buffer); }, get_variant());
}

State& State::to_(
  const torch::TensorOptions& options, bool non_blocking, bool copy)
{
  std::visit(
    [&](auto& v) { v.to_(options, non_blocking, copy); }, data_->vector_);
  return *this;
}

State& State::to_(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  std::visit(
    [&](auto& v) { v.to_(device, dtype, non_blocking, copy, memory_format); },
    data_->vector_);
  return *this;
}

State& State::to_(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  std::visit([&](auto& v) { v.to_(dtype, non_blocking, copy, memory_format); },
    data_->vector_);
  return *this;
}

State& State::to_(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  std::visit([&](auto& v) { v.to_(device, non_blocking, copy, memory_format); },
    data_->vector_);
  return *this;
}

State State::to(
  const torch::TensorOptions& options, bool non_blocking, bool copy) const
{
  auto new_variant = std::visit(
    [&](const auto& v) -> StateData::VectorVariant {
      return v.to(options, non_blocking, copy);
    },
    data_->vector_);

  return State(c10::make_intrusive<StateData>(std::move(new_variant)),
    label_.is_user_defined() ? label_.clone() : Label::create_internal());
}

State State::to(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  auto new_variant = std::visit(
    [&](const auto& v) -> StateData::VectorVariant {
      return v.to(device, dtype, non_blocking, copy);
    },
    data_->vector_);

  return State(c10::make_intrusive<StateData>(std::move(new_variant)),
    label_.is_user_defined() ? label_.clone() : Label::create_internal());
}

State State::to(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  auto new_variant = std::visit(
    [&](const auto& v) -> StateData::VectorVariant {
      return v.to(dtype, non_blocking, copy);
    },
    data_->vector_);

  return State(c10::make_intrusive<StateData>(std::move(new_variant)),
    label_.is_user_defined() ? label_.clone() : Label::create_internal());
}

State State::to(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  auto new_variant = std::visit(
    [&](const auto& v) -> StateData::VectorVariant {
      return v.to(device, non_blocking, copy);
    },
    data_->vector_);

  return State(c10::make_intrusive<StateData>(std::move(new_variant)),
    label_.is_user_defined() ? label_.clone() : Label::create_internal());
}

State& State::neg_()
{
  std::visit([](auto& v) { v.neg_(); }, data_->vector_);
  return *this;
}

State& State::narrow_(size_t dim, int64_t start, int64_t length)
{
  std::visit([dim, start, length](auto& v) { v.narrow_(dim, start, length); },
    get_variant());
  return *this;
}

State State::narrow(size_t dim, int64_t start, int64_t length) const
{
  return std::visit(
    [dim, start, length](
      const auto& v) -> State { return State(v.narrow(dim, start, length)); },
    get_variant());
}

State& State::round_(double eps, int64_t max_rank)
{
  std::visit(
    [eps, max_rank](auto& v) {
      using Type = std::decay_t<decltype(v)>;
      if constexpr (std::is_same_v<Type, TTEngine>) {
        v.round_(eps, max_rank);
      }
    },
    get_variant());

  return *this;
}

State State::round(double eps, int64_t max_rank) const
{
  return std::visit(
    [eps, max_rank](const auto& v) {
      using Type = std::decay_t<decltype(v)>;
      Type r;

      if constexpr (std::is_same_v<Type, TTEngine>) {
        r = v.round(eps, max_rank);
      } else {
        r = v;
      }

      return State(std::move(r));
    },
    get_variant());
}

torch::Tensor State::pack(const torch::Tensor& buffer) const
{
  if (!defined())
    throw utils::runtime_error(error_context("pack"), "State is undefined");
  return std::visit(
    [&buffer](const auto& v) { return v.pack(buffer); }, get_variant());
}

State State::unpack(const torch::Tensor& buffer)
{
  TORCH_CHECK(buffer.dim() == 1 && buffer.device().is_cpu(),
    "State::unpack: buffer must be a 1D CPU tensor");

  const auto format_id = static_cast<FormatType>(buffer[0].item<int64_t>());
  switch (format_id) {
  case FormatType::TENSOR_TRAIN:
    return State(TTEngine::unpack(buffer));
  default:
    throw utils::runtime_error("ttnte::linalg::State::unpack",
      "Unknown format_id in buffer: " +
        std::to_string(static_cast<int>(format_id)));
  }
}

} // namespace ttnte::linalg
