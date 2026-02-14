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

    // Basic B-Spline data
    torch::Tensor control_points;  ///< Control points (n_u1, ... n_uk, k)

private:

}
} // namespace ttnte::cad
