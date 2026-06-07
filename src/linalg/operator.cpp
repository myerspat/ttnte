#include "ttnte/linalg/operator.hpp"

namespace ttnte::linalg {

// =================================================================
// Public methods

void Operator::to_buffer(const torch::Tensor& buffer) const
{
  std::visit([&](const auto& v) { v.to_buffer(buffer); }, get_variant());
}

void Operator::from_buffer(const torch::Tensor& buffer)
{
  std::visit([&](auto& v) { v.from_buffer(buffer); }, get_variant());
}

Operator& Operator::to_(
  const torch::TensorOptions& options, bool non_blocking, bool copy)
{
  std::visit(
    [&](auto& v) { v.to_(options, non_blocking, copy); }, get_variant());
  return *this;
}

Operator& Operator::to_(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  std::visit(
    [&](auto& v) { v.to_(device, dtype, non_blocking, copy, memory_format); },
    get_variant());
  return *this;
}

Operator& Operator::to_(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  std::visit([&](auto& v) { v.to_(dtype, non_blocking, copy, memory_format); },
    get_variant());
  return *this;
}

Operator& Operator::to_(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  std::visit([&](auto& v) { v.to_(device, non_blocking, copy, memory_format); },
    get_variant());
  return *this;
}

Operator Operator::to(
  const torch::TensorOptions& options, bool non_blocking, bool copy) const
{
  auto new_variant = std::visit(
    [&](const auto& v) -> OperatorData::MatrixVariant {
      return v.to(options, non_blocking, copy);
    },
    get_variant());

  return Operator(c10::make_intrusive<OperatorData>(std::move(new_variant)),
    label_.is_user_defined() ? label_.clone() : Label::create_internal());
}

Operator Operator::to(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  auto new_variant = std::visit(
    [&](const auto& v) -> OperatorData::MatrixVariant {
      return v.to(device, dtype, non_blocking, copy);
    },
    get_variant());

  return Operator(c10::make_intrusive<OperatorData>(std::move(new_variant)),
    label_.is_user_defined() ? label_.clone() : Label::create_internal());
}

Operator Operator::to(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  auto new_variant = std::visit(
    [&](const auto& v) -> OperatorData::MatrixVariant {
      return v.to(dtype, non_blocking, copy);
    },
    get_variant());

  return Operator(c10::make_intrusive<OperatorData>(std::move(new_variant)),
    label_.is_user_defined() ? label_.clone() : Label::create_internal());
}

Operator Operator::to(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  auto new_variant = std::visit(
    [&](const auto& v) -> OperatorData::MatrixVariant {
      return v.to(device, non_blocking, copy);
    },
    get_variant());

  return Operator(c10::make_intrusive<OperatorData>(std::move(new_variant)),
    label_.is_user_defined() ? label_.clone() : Label::create_internal());
}

Operator& Operator::neg_()
{
  std::visit([](auto& v) { v.neg_(); }, get_variant());
  return *this;
}

Operator& Operator::round_(double eps, int64_t max_rank)
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

Operator Operator::round(double eps, int64_t max_rank) const
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

      return Operator(std::move(r));
    },
    get_variant());
}

} // namespace ttnte::linalg
