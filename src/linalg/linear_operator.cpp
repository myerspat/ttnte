#include "ttnte/linalg/linear_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/tt_operator.hpp"
#include <numeric>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>

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

  // Datatype and device
  const auto& dtype = operators_[0]->dtype();
  const auto& device = operators_[0]->device();

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

    if (op->dtype() != dtype || op->device() != device) {
      throw std::runtime_error(
        "All operators should be on the same device with the same data type");
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
  // Sum all results
  return std::accumulate(operators_.cbegin() + 1, operators_.cend(),
    operators_[0]->apply(x),
    [&](torch::Tensor& result, const std::shared_ptr<Operator>& op) {
      return result + op->apply(x);
    })
    .reshape({-1, 1});
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

std::shared_ptr<Operator> LinearOperator::add_(
  const std::shared_ptr<Operator>& other)
{
  throw std::runtime_error("Adding two LinearOperators is not allowed");
}

std::shared_ptr<Operator> LinearOperator::combine() const
{
  // Group all types
  std::unordered_map<std::type_index, std::vector<std::shared_ptr<Operator>>>
    grouped;
  for (const auto& op : this->operators_) {
    const Operator& opref = *op;
    grouped[typeid(opref)].push_back(op);
  }

  // Combine operators
  std::vector<std::shared_ptr<Operator>> combined_operators;
  combined_operators.reserve(grouped.size());

  for (const auto& [type, ops] : grouped) {
    combined_operators.push_back(
      (ops.size() > 1)
        ? std::reduce(std::next(ops.begin()), ops.end(), *ops.begin(),
            [](const std::shared_ptr<Operator>& a,
              const std::shared_ptr<Operator>& b) { return a->add(b); })
        : ops[0]);
  }

  // Set scale and return
  return std::make_shared<LinearOperator>(combined_operators);
}

std::shared_ptr<Operator> LinearOperator::round(const double& eps,
  const int64_t& max_rank, std::optional<int64_t> gpu_idx) const
{
  // Vector of operators
  std::vector<std::shared_ptr<Operator>> operators;
  operators.reserve(operators_.size());

  // New TTOperator
  std::shared_ptr<Operator> new_ttop = nullptr;

  for (const auto& op : operators_) {
    // Sum all TTOperators and append the rest
    if (const auto& ttop = std::dynamic_pointer_cast<TTOperator>(op)) {
      if (new_ttop == nullptr) {
        new_ttop = ttop->clone();
      } else {
        new_ttop = new_ttop->add_(ttop);
      }
    } else {
      operators.push_back(op);
    }
  }

  // Round the TTOperator and append it
  if (new_ttop != nullptr) {
    operators.push_back(std::dynamic_pointer_cast<TTOperator>(new_ttop)->round(
      eps, max_rank, gpu_idx));
  }

  // Set scale and return
  return std::make_shared<LinearOperator>(operators);
}

std::shared_ptr<Operator> LinearOperator::type(
  const caffe2::TypeMeta& dtype) const
{
  std::vector<std::shared_ptr<Operator>> ops;
  ops.reserve(operators_.size());

  for (const auto& op : operators_) {
    ops.push_back(op->type(dtype));
  }

  return std::make_shared<LinearOperator>(ops);
}

} // namespace ttnte::linalg
