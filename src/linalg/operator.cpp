#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/linear_operator.hpp"
#include <memory>

namespace ttnte::linalg {

// =================================================
// Public methods
// =================================================
std::shared_ptr<Operator> Operator::operator+(
  const std::shared_ptr<Operator>& other)
{
  // Get pointer for self
  auto self = ptr();

  // Pointer is a LinearOperator
  if (auto op = std::dynamic_pointer_cast<LinearOperator>(self)) {
    op->append(other);
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

std::shared_ptr<Operator> Operator::operator*(const double& other)
{
  // Get pointer for self
  auto self = ptr();
  self->multiply(other);
  return self;
}

} // namespace ttnte::linalg
