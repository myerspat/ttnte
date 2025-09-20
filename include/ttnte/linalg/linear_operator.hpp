#pragma once

#include "ttnte/linalg/operator.hpp"
#include <cstdint>
#include <numeric>
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
  std::shared_ptr<Operator> type(
    const caffe2::TypeMeta& dtype) const final override;
  std::shared_ptr<Operator> add_(
    const std::shared_ptr<Operator>& other) final override;
  std::shared_ptr<Operator> combine() const;
  std::shared_ptr<Operator> round(const double& eps = 1e-12,
    const int64_t& max_rank = std::numeric_limits<int64_t>::max(),
    std::optional<int64_t> gpu_idx = std::nullopt) const;

  // =================================================
  // Getters / Setters
  std::shared_ptr<Operator> clone() const final override
  {
    return std::make_shared<LinearOperator>(*this);
  }
  void set_scale(const double& scale) final override
  {
    for (auto& op : operators_) {
      op->set_scale(scale * op->scale());
    }
  }
  std::vector<std::shared_ptr<Operator>> operators() const noexcept
  {
    return operators_;
  }
  std::vector<int64_t> input_shape() const noexcept final override
  {
    const auto& shape = operators_.front()->input_shape();
    return std::vector<int64_t> {std::accumulate(
      shape.cbegin(), shape.cend(), 1, std::multiplies<int64_t> {})};
  }
  std::vector<int64_t> output_shape() const noexcept final override
  {
    const auto& shape = operators_.front()->output_shape();
    return std::vector<int64_t> {std::accumulate(
      shape.cbegin(), shape.cend(), 1, std::multiplies<int64_t> {})};
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
  torch::Device device() const noexcept final override
  {
    return operators_[0]->device();
  }
  caffe2::TypeMeta dtype() const noexcept final override
  {
    return operators_[0]->dtype();
  }
};

} // namespace ttnte::linalg
