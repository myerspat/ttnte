#pragma once

#include <c10/util/ArrayRef.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

namespace ttnte::linalg {
class ContractionStep {
private:
  // =================================================
  // Private data
  bool use_tensordot_;
  std::string einsum_expr_;
  std::vector<int64_t> ldims_;
  std::vector<int64_t> rdims_;
  std::optional<std::vector<int64_t>> permute_;

  // =================================================
  // Private methods
  std::string generate_expression(const int64_t& lndim, const int64_t& rndim,
    const std::vector<int64_t>& ldims, const std::vector<int64_t>& rdims) const;

public:
  // =================================================
  // Public constructors
  ContractionStep(const int64_t& lndim, const int64_t& rndim,
    const std::vector<int64_t>& ldims, const std::vector<int64_t>& rdims,
    const std::optional<std::vector<int64_t>>& permute);

  // =================================================
  // Public methods
  torch::Tensor contract(
    const torch::Tensor& ltensor, const torch::Tensor& rtensor) const;
};
} // namespace ttnte::linalg
