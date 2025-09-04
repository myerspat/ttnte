#pragma once

#include <ATen/ops/tensordot.h>
#include <c10/util/ArrayRef.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

namespace linalg {
class ContractionStep {
private:
  // =================================================
  // Private data
  std::vector<int64_t> ldims_;
  std::vector<int64_t> rdims_;
  std::optional<std::vector<int64_t>> permute_;

public:
  // =================================================
  // Public constructors
  ContractionStep(const std::vector<int64_t>& ldims,
    const std::vector<int64_t>& rdims,
    const std::optional<std::vector<int64_t>>& permute)
    : ldims_(ldims), rdims_(rdims), permute_(permute) {};

  // =================================================
  // Public methods
  torch::Tensor contract(
    const torch::Tensor& ltensor, const torch::Tensor& rtensor) const
  {
    // Apply tensordot and permute if needed
    return permute_.has_value()
             ? torch::permute(
                 torch::tensordot(ltensor, rtensor, ldims_, rdims_),
                 permute_.value())
             : torch::tensordot(ltensor, rtensor, ldims_, rdims_);
  };
};
} // namespace linalg
