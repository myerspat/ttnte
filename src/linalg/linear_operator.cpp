#include "ttnte/linalg/linear_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <numeric>
#include <stdexcept>

namespace ttnte::linalg {

// =================================================
// Public constructors
// =================================================
LinearOperator::LinearOperator(
  const std::vector<std::shared_ptr<Operator>>& ops)
  : operators_(ops)
{
  // Functor for calculating a product
  auto prod = [](const std::vector<int64_t>& vec) {
    return std::accumulate(
      vec.cbegin(), vec.cend(), 1, std::multiplies<int64_t> {});
  };

  // Calculate size of input and output
  const int64_t input_size = prod(operators_[0]->input_shape());
  const int64_t output_size = prod(operators_[0]->output_shape());

  for (const auto op : operators_) {
    if (input_size != prod(op->input_shape())) {
      throw std::runtime_error(
        "Size of input must be the same for each operator");
    }
    if (output_size != prod(op->output_shape())) {
      throw std::runtime_error(
        "Size of output must be the same for each operator");
    }
  }
}

// =================================================
// Public methods
// =================================================
void LinearOperator::append(const std::shared_ptr<Operator>& op)
{
  operators_.push_back(op);
}

void LinearOperator::append(const std::vector<std::shared_ptr<Operator>>& ops)
{
  operators_.insert(operators_.end(), ops.begin(), ops.end());
}

void LinearOperator::append(const LinearOperator& ops)
{
  operators_.insert(
    operators_.end(), ops.operators().begin(), ops.operators().end());
}

void LinearOperator::prepend(const std::shared_ptr<Operator>& op)
{
  operators_.insert(operators_.begin(), op);
}

torch::Tensor LinearOperator::apply(const torch::Tensor& x) const
{
  torch::InferenceMode gaurd;

  // Iterate through and sum
  const auto& result = operators_[0]->apply(x);

  // Run through the remaining
  for (size_t i = 1; i < operators_.size(); i++) {
    result.add_(operators_[i]->apply(x));
  }

  return result;
}

void LinearOperator::cuda(const int64_t idx)
{
  for (auto& op : operators_) {
    op->cuda(idx);
  }
}

void LinearOperator::cpu()
{
  for (auto& op : operators_) {
    op->cpu();
  }
}

} // namespace ttnte::linalg
