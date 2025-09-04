#include "pybind11/pybind11.h"
#include <vector>

#include "ttnte/linalg/contraction_step.hpp"
#include "ttnte/linalg/tt_operator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(linalg, m)
{
  // linalg::TTOperator
  py::class_<linalg::TTOperator>(m, "TTOperator")
    .def(py::init<const std::vector<torch::Tensor>&,
      const std::vector<linalg::ContractionStep>&, const std::vector<int64_t>&,
      const std::vector<int64_t>&>())
    .def("__matmul__", &linalg::TTOperator::matvec)
    .def("matvec", &linalg::TTOperator::matvec)
    .def_property_readonly("num_cores", &linalg::TTOperator::num_cores)
    .def_property_readonly("output_shape", &linalg::TTOperator::output_shape)
    .def_property_readonly("input_shape", &linalg::TTOperator::input_shape)
    .def_property_readonly("shape", &linalg::TTOperator::shape);

  // linalg::ContractionStep
  py::class_<linalg::ContractionStep>(m, "ContractionStep")
    .def(py::init<const std::vector<int64_t>&, const std::vector<int64_t>&,
           const std::optional<std::vector<int64_t>>&>(),
      py::arg("ldims"), py::arg("rdims"), py::arg("permute") = py::none())
    .def("contract", &linalg::ContractionStep::contract, py::arg("ltensor"),
      py::arg("rtensor"));
}
