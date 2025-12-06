#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/linear_operator.hpp"
#include <memory>

namespace ttnte::linalg {

// =================================================
// Public methods
// =================================================
Operator::~Operator() = default;

std::shared_ptr<Operator> Operator::add(
  const std::shared_ptr<Operator>& other) const
{
  // Clone
  auto op = this->clone();

  // Add in place
  return op->add_(other);
}

std::shared_ptr<Operator> Operator::operator+(
  const std::shared_ptr<Operator>& other)
{
  // Get pointer for self
  auto self = ptr();

  // Pointer is a LinearOperator
  if (auto op = std::dynamic_pointer_cast<LinearOperator>(self)) {
    if (auto other_op = std::dynamic_pointer_cast<LinearOperator>(other)) {
      op->append(other_op->operators());
    } else {
      op->append(other);
    }
    return op;
  }

  // Pointer is an not a LinearOperator but other is
  if (auto other_op = std::dynamic_pointer_cast<LinearOperator>(other)) {
    other_op->prepend(self);
    return other_op;
  }

  // Both are not LinearOperators
  return std::make_shared<LinearOperator>(
    std::vector<std::shared_ptr<Operator>> {self, other});
}

std::shared_ptr<Operator> Operator::operator-(
  const std::shared_ptr<Operator>& other)
{
  return this->operator+(other->operator-());
}

std::shared_ptr<Operator> Operator::operator-()
{
  auto copy = this->clone();
  copy->set_scale(-copy->scale());
  return copy;
}

} // namespace ttnte::linalg
