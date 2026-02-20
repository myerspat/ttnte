#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <optional>
#include "ttnte/cad/basis_functions.hpp"

namespace ttnte::cad {
class BSpline final : public BasisFunctions{
public:
    // Constructors
    BSpline();

    BSpline(torch::Tensor control_points,
          torch::Tensor knots_u,
          torch::Tensor knots_v,
          int64_t degree_u,
          int64_t degree_v);
    
    const torch::Tensor & control_points() const noexcept {
        return control_points_;
    }

private:
    // Basic B-Spline data
    torch::Tensor control_points_;  ///< Control points (n_u1, ... n_uk, k)

}
} // namespace ttnte::cad
