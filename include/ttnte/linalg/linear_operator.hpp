#pragma once

#include "ttnte/linalg/operator.hpp"
#include <cstdint>
#include <torch/extension.h>

namespace ttnte::linalg {

class LinearOperator final : public Operator {
private:
  // =================================================
  // Private data
  std::vector<std::shared_ptr<Operator>> operators_;

public:
  // =================================================
  // Public constructors
  LinearOperator(const std::vector<std::shared_ptr<Operator>>& ops);

  // =================================================
  // Public methods
  void append(const std::shared_ptr<Operator>& op);
  void append(const std::vector<std::shared_ptr<Operator>>& ops);
  void append(const LinearOperator& ops);
  void prepend(const std::shared_ptr<Operator>& op);

  torch::Tensor apply(const torch::Tensor& x) const final override;
  void cuda(const int64_t idx) final override;
  void cpu() final override;
  void multiply(const double& other) final override
  {
    for (auto& op : operators_) {
      op->multiply(other);
    }
  }

  // =================================================
  // Getters / Setters
  std::vector<std::shared_ptr<Operator>> operators() const noexcept
  {
    return operators_;
  }
  std::vector<int64_t> input_shape() const noexcept final override
  {
    return operators_.back()->input_shape();
  }
  std::vector<int64_t> output_shape() const noexcept final override
  {
    return operators_.front()->output_shape();
  }
  int64_t nelements() const noexcept final override
  {
    int64_t nelements = 0;

    for (const auto& op : operators_) {
      nelements += op->nelements();
    }

    return nelements;
  }
  double compression() const noexcept final override
  {
    int64_t total_input = 0;
    int64_t total_output = 0;

    for (size_t i = 0; i < input_shape().size(); i++) {
      total_input += input_shape()[i];
    }
    for (size_t i = 0; i < output_shape().size(); i++) {
      total_output += output_shape()[i];
    }

    return static_cast<double>(total_input * total_output) /
           static_cast<double>(nelements());
  }
};

} // namespace ttnte::linalg
