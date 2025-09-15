#pragma once

#include "ttnte/linalg/contraction_step.hpp"
#include "ttnte/linalg/operator.hpp"
#include <ATen/core/ATen_fwd.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/cuda.h>
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

namespace ttnte::linalg {
class TTOperator final : public Operator {
private:
  // =================================================
  // Private data
  std::vector<torch::Tensor> cores_;
  std::vector<ContractionStep> csteps_;
  std::vector<int64_t> lidxs_;
  std::vector<int64_t> ridxs_;

  // =================================================
  // Private methods
  std::vector<int64_t> get_shape(
    const std::size_t& i, const std::size_t& j) const;

public:
  // =================================================
  // Public constructors
  TTOperator(const std::vector<torch::Tensor>& cores,
    const std::vector<ContractionStep>& csteps,
    const std::vector<int64_t>& lidxs, const std::vector<int64_t>& ridxs);

  TTOperator(const py::object& tt);

  TTOperator(const std::vector<torch::Tensor>& cores);

  // =================================================
  // Public methods
  torch::Tensor apply(const torch::Tensor& x) const final override;
  void cuda(const int64_t idx) final override;
  void cpu() final override;
  std::shared_ptr<Operator> add_(
    const std::shared_ptr<Operator>& other) final override;
  torch::Tensor to_dense() const;

  void lr_orthogonalize(std::optional<int64_t> gpu_idx = std::nullopt);
  std::shared_ptr<TTOperator> round(const double& eps = 1e-12,
    const int64_t& max_rank = std::numeric_limits<int64_t>::max(),
    std::optional<int64_t> gpu_idx = std::nullopt);

  // =================================================
  // Getters / Setters
  std::shared_ptr<TTOperator> typed_ptr()
  {
    return std::static_pointer_cast<TTOperator>(this->ptr());
  };
  std::shared_ptr<Operator> clone() const final override
  {
    return std::make_shared<TTOperator>(*this);
  }
  std::size_t num_cores() const noexcept { return cores_.size(); };
  std::vector<torch::Tensor> cores() const noexcept { return cores_; };
  std::vector<int64_t> ranks() const noexcept
  {
    std::vector<int64_t> ranks;
    ranks.reserve(num_cores() - 1);

    for (size_t i = 1; i < num_cores(); i++) {
      ranks.push_back(cores_[i].size(0));
    }

    return ranks;
  }
  std::vector<int64_t> output_shape() const noexcept final override
  {
    return get_shape(0, 1);
  };
  std::vector<int64_t> input_shape() const noexcept final override
  {
    return get_shape(1, 2);
  };
  std::vector<std::tuple<int64_t, int64_t>> shape() const noexcept
  {
    // Initialize array
    std::vector<std::tuple<int64_t, int64_t>> shape;
    shape.reserve(cores_.size());

    // Iterate through cores
    shape.push_back(std::make_tuple(cores_[0].size(0), cores_[0].size(1)));
    for (std::size_t i = 1; i < cores_.size(); i++) {
      shape.push_back(std::make_tuple(cores_[i].size(1), cores_[i].size(2)));
    }

    return shape;
  };
  int64_t nelements() const noexcept final override
  {
    int64_t nelements = 0;

    for (const auto& core : cores_) {
      nelements += torch::numel(core);
    }

    return nelements;
  }
  double compression() const noexcept final override
  {
    int64_t full_nelements = 1;

    for (size_t i = 0; i < num_cores(); i++) {
      full_nelements *= output_shape()[i] * input_shape()[i];
    }

    return static_cast<double>(full_nelements) /
           static_cast<double>(nelements());
  }

  // =================================================
  // Getters / Setters
  friend std::ostream& operator<<(std::ostream& os, const TTOperator& op);
};
} // namespace ttnte::linalg
