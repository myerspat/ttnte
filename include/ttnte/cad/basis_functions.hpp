#pragma once

#include <optional>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace ttnte::cad {
class BasisFunctions {
public:
  // Constructors
  BasisFunctions();

  BasisFunctions(
    std::vector<torch::Tensor> knots, std::vector<int64_t> degrees);

  // Destructors
  virtual ~BasisFunctions();

  // Getters / Setters
  const std::vector<torch::Tensor>& knots() const noexcept { return knots_; }

  const std::vector<int64_t>& degrees() const noexcept { return degrees_; }

  // Evaluation Methods
  torch::Tensor find_spans(int64_t param_idx, const torch::Tensor& coords);

protected:
  // For access by derived classes
  // May potentially move private fields here at a later point

private:
  // Basic B-Spline data (in k-dimension physical space)
  const std::vector<torch::Tensor> knots_; ///< vector of knots
  const std::vector<int64_t> degrees_; ///< degrees in each parametric dimension

  // Private Helper Methods

  // Validate knots' format
  void validate_knots(torch::Tensor& new_knots);

  // Evaluate B-spline basis functions in k-dimensional space
  torch::Tensor basis_functions(
    const torch::Tensor& spans, const torch::Tensor& coords);

  // Evaluate B-spline basis function derivatives in k-dimensional space
  torch::Tensor basis_functions_ders(const torch::Tensor& spans,
    const torch::Tensor& coords, const int64_t& order);
};
} // namespace ttnte::cad
