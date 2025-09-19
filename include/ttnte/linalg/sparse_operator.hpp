#pragma once

#include "ttnte/linalg/operator.hpp"
#include <c10/core/Layout.h>
#include <cassert>
#include <torch/extension.h>
#include <tuple>

namespace ttnte::linalg {

class SparseOperator final : public Operator {
private:
  // =================================================
  // Private data
  torch::Tensor tensor_;

public:
  // =================================================
  // Public constructors
  SparseOperator(const torch::Tensor& tensor);

  // =================================================
  // Public methods
  std::shared_ptr<Operator> add_(
    const std::shared_ptr<Operator>& other) final override;

  torch::Tensor apply(const torch::Tensor& x) const final override;
  void cuda(const int64_t idx) final override;
  void cpu() final override;
  torch::Tensor to_dense() const;
  std::shared_ptr<Operator> type(
    const caffe2::TypeMeta& dtype) const final override;

  // =================================================
  // Getters / Setters
  std::shared_ptr<Operator> clone() const final override
  {
    return std::make_shared<SparseOperator>(*this);
  }
  torch::Tensor tensor() const noexcept { return tensor_; };
  std::vector<int64_t> output_shape() const noexcept final override
  {
    return {tensor_.size(0)};
  }
  std::vector<int64_t> input_shape() const noexcept final override
  {
    return {tensor_.size(1)};
  }
  std::tuple<int64_t, int64_t> shape() const noexcept
  {
    return std::make_tuple(output_shape()[0], input_shape()[0]);
  }
  int64_t nnz() const noexcept { return torch::numel(tensor_.values()); }
  int64_t nelements() const noexcept final override
  {
    if (tensor_.layout() == torch::kSparse) {
      return nnz() + torch::numel(tensor_.indices());
    } else {
      return nnz() + torch::numel(tensor_.crow_indices()) +
             torch::numel(tensor_.col_indices());
    }
  }
  double compression() const noexcept final override
  {
    return static_cast<double>(torch::numel(tensor_)) /
           static_cast<double>(nelements());
  }
  bool is_sparse() const noexcept { return true; }
  torch::Device device() const noexcept final override
  {
    return tensor_.device();
  }
  caffe2::TypeMeta dtype() const noexcept final override
  {
    return tensor_.dtype();
  }
};

} // namespace ttnte::linalg
