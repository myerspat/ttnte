#include "ttnte/cad/basis_functions.hpp"
#include <optional>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

void register_BasisFunctions(py::module_& m)
{
  using BasisFunctions = ttnte::cad::BasisFunctions;

  py::class_<BasisFunctions, std::shared_ptr<BasisFunctions>>(
    m, "BasisFunctions")
    .def(
      py::init<const std::vector<torch::Tensor>&, const std::vector<int64>&>(),
      py::arg("knots"), py::arg("degree"))
    .def("find_spans", &BasisFunctions::find_spans)
    .def_readwrite("knots", &BasisFunctions::knots)
    .def_readwrite("degree", &BasisFunctions::degree)
}
