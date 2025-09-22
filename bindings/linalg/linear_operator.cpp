#include "ttnte/linalg/linear_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object caffe2torch(const caffe2::TypeMeta& dtype);
caffe2::TypeMeta torch2caffe(const py::object& dtype);

void register_LinearOperator(py::module_& m)
{
  using Operator = ttnte::linalg::Operator;
  using LinearOperator = ttnte::linalg::LinearOperator;

  py::class_<LinearOperator, std::shared_ptr<LinearOperator>, Operator>(
    m, "LinearOperator")
    .def(py::init<const std::vector<std::shared_ptr<Operator>>&>())
    .def("__matmul__", &LinearOperator::apply)
    .def("matvec", &LinearOperator::apply)
    .def("apply", &LinearOperator::apply)
    .def("cuda", &LinearOperator::cuda)
    .def("cpu", &LinearOperator::cpu)
    .def("combine", &LinearOperator::combine)
    .def("round", &LinearOperator::round, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::arg("gpu_idx") = std::nullopt)
    .def("clone", &LinearOperator::clone)
    .def("type",
      [](const LinearOperator& self, const py::object& dtype) {
        return self.type(torch2caffe(dtype));
      })
    .def_property_readonly("operators", &LinearOperator::operators)
    .def_property_readonly("nelements", &LinearOperator::nelements)
    .def_property_readonly("compression", &LinearOperator::compression)
    .def_property_readonly("device",
      [](const LinearOperator& self) { return py::cast(self.device()); })
    .def_property_readonly("dtype",
      [](const LinearOperator& self) { return caffe2torch(self.dtype()); });
}
