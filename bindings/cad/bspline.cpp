#include "ttnte/cad/bspline.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_bspline(py::module_& m)
{
    using bspline = ttnte::cad::bspline;

    py::class_<bspline>(m, "bspline", R"pbdoc(
        NURBS bspline for modelling and parametric surface operations.
    )pbdoc")

        // ====================================================================
        // Constructors
        .def(py::init<>())

        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor,
                     int64_t, int64_t>(),
             py::arg("control_points"),
             py::arg("knots_u"),
             py::arg("knots_v"),
             py::arg("degree_u"),
             py::arg("degree_v"))

}
