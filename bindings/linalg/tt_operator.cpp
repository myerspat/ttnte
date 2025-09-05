#include "ttnte/linalg/tt_operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_TTOperator(py::module_& m)
{
  py::class_<ttnte::linalg::TTOperator>(m, "TTOperator")
    .def(py::init<const std::vector<torch::Tensor>&,
      const std::vector<ttnte::linalg::ContractionStep>&,
      const std::vector<int64_t>&, const std::vector<int64_t>&>())
    .def(py::init<const py::object&>())
    .def("__matmul__", &ttnte::linalg::TTOperator::matvec)
    .def("matvec", &ttnte::linalg::TTOperator::matvec)
    .def("cuda", &ttnte::linalg::TTOperator::cuda)
    .def("cpu", &ttnte::linalg::TTOperator::cpu)
    .def_property_readonly("num_cores", &ttnte::linalg::TTOperator::num_cores)
    .def_property_readonly("cores", &ttnte::linalg::TTOperator::cores)
    .def_property_readonly(
      "output_shape", &ttnte::linalg::TTOperator::output_shape)
    .def_property_readonly(
      "input_shape", &ttnte::linalg::TTOperator::input_shape)
    .def_property_readonly("shape", &ttnte::linalg::TTOperator::shape)
    .def_property_readonly("nelements", &ttnte::linalg::TTOperator::nelements)
    .def_property_readonly(
      "compression", &ttnte::linalg::TTOperator::compression);
}
