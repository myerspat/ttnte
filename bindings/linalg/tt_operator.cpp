#include "ttnte/linalg/tt_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_TTOperator(py::module_& m)
{
  using Operator = ttnte::linalg::Operator;
  using TTOperator = ttnte::linalg::TTOperator;
  using ContractionStep = ttnte::linalg::ContractionStep;

  // Get python version for docstrings
  const auto& pyclass =
    py::module::import("ttnte.linalg.tt_operator").attr("TTOperator");

  py::class_<TTOperator, std::shared_ptr<TTOperator>, Operator>(m, "TTOperator")
    .def(py::init<const std::vector<torch::Tensor>&,
      const std::vector<ContractionStep>&, const std::vector<int64_t>&,
      const std::vector<int64_t>&>())
    .def(py::init<const py::object&>())
    .def("__matmul__", &TTOperator::apply)
    .def("matvec", &TTOperator::apply)
    .def("apply", &TTOperator::apply)
    .def("cuda", &TTOperator::cuda)
    .def("cpu", &TTOperator::cpu)
    .def("lr_orthogonalize", &TTOperator::lr_orthogonalize)
    .def("round", &TTOperator::round, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::arg("gpu_idx") = std::nullopt)
    .def("to_dense", &TTOperator::to_dense)
    .def_property_readonly("num_cores", &TTOperator::num_cores)
    .def_property_readonly("cores", &TTOperator::cores)
    .def_property_readonly("output_shape", &TTOperator::output_shape)
    .def_property_readonly("input_shape", &TTOperator::input_shape)
    .def_property_readonly("shape", &TTOperator::shape)
    .def_property_readonly("ranks", &TTOperator::ranks)
    .def_property_readonly("nelements", &TTOperator::nelements)
    .def_property_readonly("compression", &TTOperator::compression)
    .doc() = pyclass.attr("__doc__");
}
