#include "ttnte/linalg/fission_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_FissionOperator(py::module_& m)
{
  using Operator = ttnte::linalg::Operator;
  using FissionOperator = ttnte::linalg::FissionOperator;

  py::class_<FissionOperator, std::shared_ptr<FissionOperator>, Operator>(
    m, "FissionOperator")
    .def(py::init<const torch::Tensor&, const torch::Tensor&,
      const torch::Tensor&>())
    .def("__matmul__", &FissionOperator::apply)
    .def("matvec", &FissionOperator::apply)
    .def("apply", &FissionOperator::apply)
    .def("cuda", &FissionOperator::cuda)
    .def("cpu", &FissionOperator::cpu)
    .def("clone", &FissionOperator::clone)
    .def_property_readonly("F", &FissionOperator::F)
    .def_property_readonly("output_shape", &FissionOperator::output_shape)
    .def_property_readonly("input_shape", &FissionOperator::input_shape)
    .def_property_readonly("shape", &FissionOperator::shape)
    .def_property_readonly("nelements", &FissionOperator::nelements)
    .def_property_readonly("compression", &FissionOperator::compression);
}
