#include "ttnte/linalg/operator.hpp"
#include "../utils/label.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_Operator(py::module_& m)
{
  using namespace ttnte::linalg;

  register_Label<Operator>(m, "Operator");
  py::class_<Operator> op(
    m, "Operator", "ttnte linear algebra operator matrix");

  // =================================================================
  // Constructors (Keep GIL: Fast allocations / string conversions)
  op.def(py::init<>(), "Creates an undefined operator")
    .def(py::init<const Operator&>(), py::arg("other"), "Copy constructor")
    .def(py::init<TTEngine, std::optional<std::string>>(), py::arg("engine"),
      py::arg("label") = py::none(), "Construct from TTEngine");

  // =================================================================
  // Properties & Getters/Setters (Keep GIL: Trivial O(1) accesses)
  op.def_property(
      "label", [](const Operator& o) { return o.get_label(); },
      &Operator::set_label)
    .def_property_readonly("is_cuda", &Operator::is_cuda)
    .def_property_readonly("device", &Operator::get_device)
    .def_property_readonly("dtype", &Operator::get_dtype)
    .def_property_readonly("numel", &Operator::get_numel)
    .def_property_readonly("is_tt", &Operator::is_tt);

  // =================================================================
  // Core Methods (Release GIL: High-latency compute and memory IO)
  op.def("defined", &Operator::defined,
      "Check if the underlying data is allocated")
    .def("as_tt", &Operator::as_tt, py::return_value_policy::reference_internal)

    .def("to_buffer", &Operator::to_buffer, py::arg("buffer"),
      py::call_guard<py::gil_scoped_release>())
    .def("from_buffer", &Operator::from_buffer, py::arg("buffer"),
      py::call_guard<py::gil_scoped_release>())
    .def("neg_", &Operator::neg_, py::call_guard<py::gil_scoped_release>())

    .def("round_", &Operator::round_, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())
    .def("round", &Operator::round, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // Device & Dtype Transfers (to / to_ with GIL Released)

  // In-place to_()
  op.def("to_",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &Operator::to_),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  // Out-of-place to()
  op.def("to",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &Operator::to, py::const_),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // Python Magic Methods for Math (Release GIL)

  // Unary Negation
  op.def(
    "__neg__", [](const Operator& a) { return -a; },
    py::call_guard<py::gil_scoped_release>());

  // Addition
  op.def(
      "__add__", [](const Operator& a, const Operator& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__", [](const Operator& a, double b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__",
      [](const Operator& a, const torch::Tensor& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__", [](const Operator& a, double b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__",
      [](const Operator& a, const torch::Tensor& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>());

  // Subtraction
  op.def(
      "__sub__", [](const Operator& a, const Operator& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__", [](const Operator& a, double b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__",
      [](const Operator& a, const torch::Tensor& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__", [](const Operator& a, double b) { return b - a; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__",
      [](const Operator& a, const torch::Tensor& b) { return b - a; },
      py::call_guard<py::gil_scoped_release>());

  // Multiplication
  op.def(
      "__mul__", [](const Operator& a, const Operator& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__mul__", [](const Operator& a, double b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__mul__",
      [](const Operator& a, const torch::Tensor& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rmul__", [](const Operator& a, double b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rmul__",
      [](const Operator& a, const torch::Tensor& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>());

  // Division
  op.def(
      "__truediv__", [](const Operator& a, const Operator& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__truediv__", [](const Operator& a, double b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__truediv__",
      [](const Operator& a, const torch::Tensor& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rtruediv__", [](const Operator& a, double b) { return b / a; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rtruediv__",
      [](const Operator& a, const torch::Tensor& b) { return b / a; },
      py::call_guard<py::gil_scoped_release>());

  // In-place Operations
  op.def(
      "__iadd__",
      [](Operator& a, const Operator& b) -> Operator& {
        a += b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__iadd__",
      [](Operator& a, double b) -> Operator& {
        a += b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__iadd__",
      [](Operator& a, const torch::Tensor& b) -> Operator& {
        a += b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())

    .def(
      "__isub__",
      [](Operator& a, const Operator& b) -> Operator& {
        a -= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__isub__",
      [](Operator& a, double b) -> Operator& {
        a -= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__isub__",
      [](Operator& a, const torch::Tensor& b) -> Operator& {
        a -= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())

    .def(
      "__imul__",
      [](Operator& a, const Operator& b) -> Operator& {
        a *= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__imul__",
      [](Operator& a, double b) -> Operator& {
        a *= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__imul__",
      [](Operator& a, const torch::Tensor& b) -> Operator& {
        a *= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())

    .def(
      "__itruediv__",
      [](Operator& a, const Operator& b) -> Operator& {
        a /= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__itruediv__",
      [](Operator& a, double b) -> Operator& {
        a /= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__itruediv__",
      [](Operator& a, const torch::Tensor& b) -> Operator& {
        a /= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>());
}
