#include "ttnte/linalg/linear_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_LinearOperator(py::module_& m)
{
  using Operator = ttnte::linalg::Operator;
  using LinearOperator = ttnte::linalg::LinearOperator;

  py::class_<LinearOperator, std::shared_ptr<LinearOperator>, Operator>(
    m, "LinearOperator")
    .def(py::init<const std::vector<std::shared_ptr<Operator>>&>())
    .def("combine", &LinearOperator::combine)
    .def("round", &LinearOperator::round, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::arg("gpu_idx") = std::nullopt)
    .def_property_readonly("operators", &LinearOperator::operators)
    .def_property_readonly("nelements", &LinearOperator::nelements)
    .def_property_readonly("compression", &LinearOperator::compression);
}
