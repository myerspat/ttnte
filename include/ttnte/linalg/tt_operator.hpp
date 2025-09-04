#pragma once

#include "ttnte/linalg/contraction_step.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

namespace linalg {
class TTOperator {
private:
  // =================================================
  // Private data
  std::vector<torch::Tensor> cores_;
  std::vector<ContractionStep> csteps_;
  std::vector<int64_t> lidxs_;
  std::vector<int64_t> ridxs_;

  // =================================================
  // Private methods
  std::vector<std::size_t> get_shape(
    const std::size_t& i, const std::size_t& j) const
  {
    // Initialize shape
    std::vector<std::size_t> shape;
    shape.reserve(cores_.size());

    // Add shapes
    shape.push_back(cores_[0].size(i));
    for (size_t k = 1; k < cores_.size(); k++) {
      shape.push_back(cores_[k].size(j));
    }

    return shape;
  }

public:
  // =================================================
  // Public constructors
  TTOperator(const std::vector<torch::Tensor>& cores,
    const std::vector<ContractionStep>& csteps,
    const std::vector<int64_t>& lidxs, const std::vector<int64_t>& ridxs)
    : cores_(cores), csteps_(csteps), lidxs_(lidxs), ridxs_(ridxs)
  {
    // Check lengths
    assert(csteps.size == lidxs.size);
    assert(csteps.size == ridxs.size);
  };

  // =================================================
  // Public overloads
  torch::Tensor matvec(const torch::Tensor& x) const
  {
    // Create tensor bank with all tensors
    std::vector<torch::Tensor> tensor_bank = cores_;
    tensor_bank.emplace_back(x);

    // Loop through contraction steps
    for (size_t i = 0; i < csteps_.size(); ++i) {
      // Get tensors
      const auto& ltensor = tensor_bank[lidxs_[i]];
      const auto& rtensor = tensor_bank[ridxs_[i]];

      // Erase outer tensor
      tensor_bank.erase(tensor_bank.begin() + std::max(lidxs_[i], ridxs_[i]));

      // Check that the tensors are on the same device
      assert(ltensor.device() == rtensor.device());

      // Compute contractions and place in left tensor position
      tensor_bank[std::min(lidxs_[i], ridxs_[i])] =
        csteps_[i].contract(ltensor, rtensor);
    }

    // Check there is one tensor left in tensor bank
    assert(tensor_bank.size == 1);
    return tensor_bank[0];
  };

  // =================================================
  // Getters / Setters
  std::size_t num_cores() const noexcept { return cores_.size(); };
  std::vector<std::size_t> output_shape() const noexcept
  {
    return get_shape(0, 1);
  };
  std::vector<std::size_t> input_shape() const noexcept
  {
    return get_shape(1, 2);
  };
  std::vector<std::tuple<std::size_t, std::size_t>> shape() const noexcept
  {
    // Initialize array
    std::vector<std::tuple<std::size_t, std::size_t>> shape;
    shape.reserve(cores_.size());

    // Iterate through cores
    shape.push_back(std::make_tuple(cores_[0].size(0), cores_[0].size(1)));
    for (std::size_t i = 1; i < cores_.size(); i++) {
      shape.push_back(std::make_tuple(cores_[i].size(1), cores_[i].size(2)));
    }

    return shape;
  };
};
} // namespace linalg
