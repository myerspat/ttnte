#include <ttnte/cad/bspline.hpp>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <optional>

namespace ttnte::cad {

    BSpline::BSpline()
    : degree_u(0)
    , degree_v(0)
    {
    }

    Patch::Patch(torch::Tensor control_points,
             torch::Tensor knots_u,
             torch::Tensor knots_v,
             int64_t degree_u,
             int64_t degree_v)
    : control_points(control_points)
    , knots_u(knots_u)
    , knots_v(knots_v)
    , degree_u(degree_u)
    , degree_v(degree_v)
    {
        // Validate inputs
        if (control_points.dim() != 3 || control_points.size(2) != 3) {
            throw std::invalid_argument("control_points must have shape (n_u, n_v, 3)");
        }
        if (knots_u.dim() != 1 || knots_v.dim() != 1) {
            throw std::invalid_argument("knot vectors must be 1-dimensional");
        }
    }

} // namespace ttnte::cad
