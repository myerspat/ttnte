#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/torch.h>
#include <torch/types.h>

namespace py = pybind11;

py::object caffe2torch(const caffe2::TypeMeta& dtype)
{
  // Lookup Python torch module
  py::module t = py::module::import("torch");

  if (dtype == torch::kFloat64)
    return t.attr("float64");
  if (dtype == torch::kFloat32)
    return t.attr("float32");
  if (dtype == torch::kFloat16)
    return t.attr("float16");

  // Default case
  return py::str(std::string(dtype.name().substr()));
}

caffe2::TypeMeta torch2caffe(const py::object& dtype)
{
  // Lookup Python torch module
  py::module t = py::module::import("torch");

  if (dtype.is(t.attr("float64")))
    return torch::dtype(torch::kFloat64).dtype();
  if (dtype.is(t.attr("float32")))
    return torch::dtype(torch::kFloat32).dtype();
  if (dtype.is(t.attr("float16")))
    return torch::dtype(torch::kFloat16).dtype();

  throw std::runtime_error("Type is not supported");
}
