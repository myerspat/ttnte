#pragma once

#include <c10/core/Layout.h>
#include <cassert>
#include <torch/extension.h>
#include <tuple>

namespace ttnte::linalg {

class CSROperator {
private:
  // =================================================
  // Private data
  torch::Tensor tensor_;

public:
  // =================================================
  // Public constructors
  CSROperator(const torch::Tensor& tensor);

  // =================================================
  // Public methods
  torch::Tensor matvec(const torch::Tensor& x) const;

  void cuda(const int64_t idx);

  void cpu();

  // =================================================
  // Getters / Setters
  torch::Tensor tensor() const noexcept { return tensor_; };
  int64_t output_shape() const noexcept { return tensor_.size(0); }
  int64_t input_shape() const noexcept { return tensor_.size(1); }
  std::tuple<std::size_t, std::size_t> shape() const noexcept
  {
    return std::make_tuple(output_shape(), input_shape());
  }
  int64_t nnz() const noexcept
  {
    assert(tensor_.layout() == torch::kSparseCsr);
    return torch::numel(tensor_.values());
  }
  int64_t nelements() const noexcept
  {
    assert(tensor_.layout() == torch::kSparseCsr);
    return nnz() + torch::numel(tensor_.crow_indices()) +
           torch::numel(tensor_.col_indices());
  }
  double compression() const noexcept
  {
    return static_cast<double>(torch::numel(tensor_)) /
           static_cast<double>(nelements());
  }
  bool is_sparse() const noexcept { return true; }
};

} // namespace ttnte::linalg
