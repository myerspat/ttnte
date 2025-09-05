#include "ttnte/linalg/csr_operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_CSROperator(py::module_& m)
{
  using CSROperator = ttnte::linalg::CSROperator;

  py::class_<CSROperator>(m, "CSROperator")
    .def(py::init<const torch::Tensor&>())
    .def("__matmul__", &CSROperator::matvec)
    .def("matvec", &CSROperator::matvec)
    .def("cuda", &CSROperator::cuda)
    .def("cpu", &CSROperator::cpu)
    .def_property_readonly("tensor", &CSROperator::tensor)
    .def_property_readonly("output_shape", &CSROperator::output_shape)
    .def_property_readonly("input_shape", &CSROperator::input_shape)
    .def_property_readonly("shape", &CSROperator::shape)
    .def_property_readonly("nnz", &CSROperator::nnz)
    .def_property_readonly("nelements", &CSROperator::nelements)
    .def_property_readonly("compression", &CSROperator::compression);
}
