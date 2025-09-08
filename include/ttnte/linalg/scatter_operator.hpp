#pragma once

#include "ttnte/linalg/operator.hpp"
#include <torch/extension.h>

namespace ttnte::linalg {

class ScatterOperator final : public Operator {
private:
  // =================================================
  // Private data
  std::vector<torch::Tensor> S_;
  torch::Tensor Y_;
  torch::Tensor w_mu_;
  torch::Tensor w_eta_;

public:
  // =================================================
  // Public constructors
  ScatterOperator(const std::vector<torch::Tensor>& S, const torch::Tensor& Y,
    const torch::Tensor& w_mu, const torch::Tensor& w_eta);

  // =================================================
  // Public methods
  torch::Tensor apply(const torch::Tensor& x) const final override;
  void cuda(const int64_t idx) final override;
  void cpu() final override;
  void multiply(const double& other) final override { w_mu_ *= other; }

  // =================================================
  // Getters / Setters
  std::vector<torch::Tensor> S() const noexcept { return S_; }
  torch::Tensor Y() const noexcept { return Y_; }
  std::vector<int64_t> output_shape() const noexcept final override
  {
    return {Y_.size(1) * Y_.size(2) * Y_.size(3) * S_[0].size(0)};
  };
  std::vector<int64_t> input_shape() const noexcept final override
  {
    return output_shape();
  };
  std::tuple<int64_t, int64_t> shape() const noexcept
  {
    return std::make_tuple(output_shape()[0], input_shape()[0]);
  }
  int64_t nelements() const noexcept final override
  {
    int64_t nelements = 0;

    for (const auto& s : S_) {
      if (!s.is_sparse()) {
        nelements += torch::numel(s);
      } else if (s.layout() == torch::kSparseCsr) {
        nelements += torch::numel(s.values()) + torch::numel(s.crow_indices()) +
                     torch::numel(s.col_indices());
      } else {
        nelements += torch::numel(s.values()) + torch::numel(s.indices());
      }
    }

    return nelements + torch::numel(Y_) + torch::numel(w_mu_) +
           torch::numel(w_eta_);
  };
  double compression() const noexcept final override
  {
    return static_cast<double>(output_shape()[0] * input_shape()[0]) /
           static_cast<double>(nelements());
  }
};

} // namespace ttnte::linalg
