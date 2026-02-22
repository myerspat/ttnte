#pragma once
#include "ttnte/cad/basis_functions.hpp"

#include <optional>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace ttnte::cad {
class Bspline : public BasisFunctions {
public:
  // Constructors
  Bspline();

  Bspline(std::vector<torch::Tensor> knots,
    std::vector<torch::Tensor> ctrl_pts_, std::vector<int64_t> degrees);

  // Destructors
  virtual ~Bspline();

  // Getters / Setters
  const std::vector<int64_t>& ctrl_pts() const noexcept { return ctrl_pts_; }

  // Evaluation Methods
  // Knot Refinement
  void knot_refine(int64_t param_idx, const torch::Tensor& ctrl_pts,
    const torch::Tensor& knots);

protected:
  // For access by derived classes
  // May potentially move private fields here at a later point

private:
  // Basic B-Spline data (in k-dimension physical space)
  const std::vector<torch::Tensor>
    ctrl_pts_; ///< vector of ctrl pts in each dimension

  // Private Helper Methods
};
} // namespace ttnte::cad
