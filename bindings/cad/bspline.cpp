#include "ttnte/cad/bspline.hpp"
#include "ttnte/cad/basis_functions.hpp"
#include <optional>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

void register_Bspline(py::module_& m)
{
  using Bspline = ttnte::cad::Bspline;

  py::class_<Bspline, BasisFunctions>(m, "Bspline")
    .def(py::init<const std::vector<torch::Tensor>&,
           const std::vector<int64_t>&>(),
      const std::vector<torch::Tensor>& py::arg("knots"), py::arg("degrees")),
    py::arg("ctrl_pts")
      .def("knot_refine", &Bspline::knot_refine)
      .def_property_readonly("ctrl_pts", &Bspline::ctrl_pts);
}
