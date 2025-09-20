#include "ttnte/linalg/scatter_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object caffe2torch(const caffe2::TypeMeta& dtype);
caffe2::TypeMeta torch2caffe(const py::object& dtype);

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
    .def("type",
      [](const ScatterOperator& self, const py::object& dtype) {
        return self.type(torch2caffe(dtype));
      })
    .def_property_readonly("S", &ScatterOperator::S)
    .def_property_readonly("Y", &ScatterOperator::Y)
    .def_property_readonly("output_shape", &ScatterOperator::output_shape)
    .def_property_readonly("input_shape", &ScatterOperator::input_shape)
    .def_property_readonly("shape", &ScatterOperator::shape)
    .def_property_readonly("nelements", &ScatterOperator::nelements)
    .def_property_readonly("compression", &ScatterOperator::compression)
    .def_property_readonly("device",
      [](const ScatterOperator& self) { return py::cast(self.device()); })
    .def_property_readonly("dtype",
      [](const ScatterOperator& self) { return caffe2torch(self.dtype()); });
}
