#include "ttnte/linalg/scatter_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_ScatterOperator(py::module_& m)
{
  using Operator = ttnte::linalg::Operator;
  using ScatterOperator = ttnte::linalg::ScatterOperator;

  py::class_<ScatterOperator, std::shared_ptr<ScatterOperator>, Operator>(
    m, "ScatterOperator")
    .def(py::init<const std::vector<torch::Tensor>&, const torch::Tensor&,
      const torch::Tensor&, const torch::Tensor&>())
    .def("__matmul__", &ScatterOperator::apply)
    .def("matvec", &ScatterOperator::apply)
    .def("apply", &ScatterOperator::apply)
    .def("cuda", &ScatterOperator::cuda)
    .def("cpu", &ScatterOperator::cpu)
    .def("clone", &ScatterOperator::clone)
    .def_property_readonly("S", &ScatterOperator::S)
    .def_property_readonly("Y", &ScatterOperator::Y)
    .def_property_readonly("output_shape", &ScatterOperator::output_shape)
    .def_property_readonly("input_shape", &ScatterOperator::input_shape)
    .def_property_readonly("shape", &ScatterOperator::shape)
    .def_property_readonly("nelements", &ScatterOperator::nelements)
    .def_property_readonly("compression", &ScatterOperator::compression);
}
