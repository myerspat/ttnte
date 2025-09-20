#include "ttnte/linalg/fission_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object caffe2torch(const caffe2::TypeMeta& dtype);
caffe2::TypeMeta torch2caffe(const py::object& dtype);

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
    .def("type",
      [](const FissionOperator& self, const py::object& dtype) {
        return self.type(torch2caffe(dtype));
      })
    .def_property_readonly("F", &FissionOperator::F)
    .def_property_readonly("output_shape", &FissionOperator::output_shape)
    .def_property_readonly("input_shape", &FissionOperator::input_shape)
    .def_property_readonly("shape", &FissionOperator::shape)
    .def_property_readonly("nelements", &FissionOperator::nelements)
    .def_property_readonly("compression", &FissionOperator::compression)
    .def_property_readonly("device",
      [](const FissionOperator& self) { return py::cast(self.device()); })
    .def_property_readonly("dtype",
      [](const FissionOperator& self) { return caffe2torch(self.dtype()); });
}
