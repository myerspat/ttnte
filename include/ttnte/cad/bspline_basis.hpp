#pragma once

#include "ttnte/cad/basis.hpp"
#include <ATen/ops/searchsorted.h>
#include <cstdint>
#include <ostream>
#include <torch/torch.h>

namespace ttnte::cad {

class BSplineBasis : public Basis<BSplineBasis> {
public:
  // =================================================================
  // Public data
  static constexpr const char* class_name = "BSplineBasis";

private:
  // =================================================================
  // Private data
  torch::Tensor knotvector_;
  int64_t degree_;

public:
  // =================================================================
  // Public constructors
  BSplineBasis(const torch::Tensor& knotvector, int64_t degree,
    std::optional<std::string> label = std::nullopt);
  BSplineBasis(
    const torch::Tensor& knotvector, int64_t degree, const Label& label);

  // =================================================================
  // Public methods
  bool is_valid() const;
  void normalize_knotvector();
  BSplineBasis& to_(std::optional<torch::Device> device,
    std::optional<torch::ScalarType> dtype);
  BSplineBasis to(std::optional<torch::Device> device,
    std::optional<torch::ScalarType> dtype) const;
  BSplineBasis clone() const;

  // // TODO: Find the first index where the interval begins for u
  // // For example if x = 0.5 and knotvector_ = [0, 0, 0, 1, 1, 1]
  // // then we return 2
  // int64_t inline find_span(const double& u) const
  // {
  //   throw utils::runtime_error(
  //     *this, error_context("find_span"), "Not implemented yet");
  // }

  torch::Tensor find_spans(const torch::Tensor& u) const;

  // // TODO: Function that takes in a single double between [0, 1]
  // // and returns a torch::Tensor of non-zero basis values at x
  // // If derivative_order is non-zero then evaluate the derivative there
  // torch::Tensor inline evaluate(
  //   const double& u, const int64_t& derivative_order = 0)
  // {
  //   throw utils::runtime_error(
  //     *this, error_context("evaluate"), "Not implemented yet");
  // }

  torch::Tensor evaluate(
    const torch::Tensor& u, const int64_t& derivative_order = 0) const;

  torch::Tensor evaluate(const torch::Tensor& u, const torch::Tensor& spans,
    const int64_t& derivative_order = 0) const;

  torch::Tensor evaluate_all(
    const torch::Tensor& u, const int64_t& derivative_order = 0) const;

  torch::Tensor evaluate_all(const torch::Tensor& u, const torch::Tensor& spans,
    const int64_t& derivative_order = 0) const;

  // =================================================================
  // Public getters
  const inline int64_t& get_degree() const noexcept { return degree_; }
  const inline torch::Tensor& get_knotvector() const noexcept
  {
    return knotvector_;
  }
  inline int64_t get_order() const noexcept { return degree_ + 1; }
  inline std::tuple<torch::Tensor, torch::Tensor>
  get_unique_knots_and_multiplicity() const noexcept
  {
    auto result = torch::unique_consecutive(knotvector_, false, true);
    return std::make_tuple(std::get<0>(result), std::get<2>(result));
  }
  inline torch::Tensor get_unique_knots() const noexcept
  {
    return std::get<0>(get_unique_knots_and_multiplicity());
  }
  inline torch::Tensor get_multiplicity() const noexcept
  {
    return std::get<1>(get_unique_knots_and_multiplicity());
  }
  inline int64_t get_size() const noexcept
  {
    return knotvector_.size(0) - degree_ - 1;
  }
  inline torch::Device get_device() const noexcept
  {
    return knotvector_.device();
  }
  inline torch::ScalarType get_dtype() const noexcept
  {
    return knotvector_.scalar_type();
  }
};

std::ostream& operator<<(std::ostream& os, const BSplineBasis& p);

} // namespace ttnte::cad
