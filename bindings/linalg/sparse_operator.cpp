#include "ttnte/linalg/sparse_operator.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object caffe2torch(const caffe2::TypeMeta& dtype);
caffe2::TypeMeta torch2caffe(const py::object& dtype);

void register_SparseOperator(py::module_& m)
{
  using Operator = ttnte::linalg::Operator;
  using SparseOperator = ttnte::linalg::SparseOperator;

  py::class_<SparseOperator, std::shared_ptr<SparseOperator>, Operator>(
    m, "SparseOperator")
    .def(py::init<const torch::Tensor&>())
    .def("__matmul__", &SparseOperator::apply)
    .def("matvec", &SparseOperator::apply)
    .def("apply", &SparseOperator::apply)
    .def("cuda", &SparseOperator::cuda)
    .def("cpu", &SparseOperator::cpu)
    .def("to_dense", &SparseOperator::to_dense)
    .def("clone", &SparseOperator::clone)
    .def("type",
      [](const SparseOperator& self, const py::object& dtype) {
        return self.type(torch2caffe(dtype));
      })
    .def_property_readonly("tensor", &SparseOperator::tensor)
    .def_property_readonly("output_shape", &SparseOperator::output_shape)
    .def_property_readonly("input_shape", &SparseOperator::input_shape)
    .def_property_readonly("shape", &SparseOperator::shape)
    .def_property_readonly("nnz", &SparseOperator::nnz)
    .def_property_readonly("nelements", &SparseOperator::nelements)
    .def_property_readonly("compression", &SparseOperator::compression)
    .def_property_readonly("device",
      [](const SparseOperator& self) { return py::cast(self.device()); })
    .def_property_readonly("dtype",
      [](const SparseOperator& self) { return caffe2torch(self.dtype()); });
}
