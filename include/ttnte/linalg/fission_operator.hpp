#pragma once

#include "ttnte/linalg/operator.hpp"
#include <torch/extension.h>

namespace ttnte::linalg {

class FissionOperator final : public Operator {
private:
  // =================================================
  // Private data
  torch::Tensor F_;
  torch::Tensor w_mu_;
  torch::Tensor w_eta_;

public:
  // =================================================
  // Public constructors
  FissionOperator(const torch::Tensor& F, const torch::Tensor& w_mu,
    const torch::Tensor& w_eta);

  // =================================================
  // Public methods
  torch::Tensor apply(const torch::Tensor& x) const final override;
  void cuda(const int64_t idx) final override;
  void cpu() final override;
  void multiply(const double& other) final override { w_mu_ *= other; }

  // =================================================
  // Getters / Setters
  torch::Tensor F() const noexcept { return F_; };
  std::vector<int64_t> output_shape() const noexcept final override
  {
    return {4 * w_mu_.size(0) * w_eta_.size(0) * F_.size(0)};
  }
  std::vector<int64_t> input_shape() const noexcept final override
  {
    return output_shape();
  }
  std::tuple<int64_t, int64_t> shape() const noexcept
  {
    return std::make_tuple(output_shape()[0], input_shape()[0]);
  }
  int64_t nelements() const noexcept final override
  {
    int64_t nelements;

    if (!F_.is_sparse()) {
      nelements += torch::numel(F_);
    } else if (F_.layout() == torch::kSparseCsr) {
      nelements += torch::numel(F_.values()) + torch::numel(F_.crow_indices()) +
                   torch::numel(F_.col_indices());
    } else {
      nelements += torch::numel(F_.values()) + torch::numel(F_.indices());
    }

    return nelements + torch::numel(w_mu_) + torch::numel(w_eta_);
  }
  double compression() const noexcept final override
  {
    return static_cast<double>(output_shape()[0] * input_shape()[0]) /
           static_cast<double>(nelements());
  }
};

} // namespace ttnte::linalg
