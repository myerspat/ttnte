#include "ttnte/linalg/state.hpp"
#include "../utils/label.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_State(py::module_& m)
{
  using namespace ttnte::linalg;

  register_Label<State>(m, "State");
  py::class_<State> state(m, "State", "TTNTE linear algebra state vector");

  // =================================================================
  // Constructors (Keep GIL: Fast operations / string conversions)
  state.def(py::init<>(), "Creates an undefined state")
    .def(py::init<const State&>(), py::arg("other"), "Copy constructor")
    .def(py::init<TTEngine, std::optional<std::string>>(), py::arg("engine"),
      py::arg("label") = py::none(), "Construct from TTEngine");

  // =================================================================
  // Properties & Getters/Setters (Keep GIL: O(1) accesses)
  state
    .def_property(
      "label", [](const State& s) { return s.get_label(); }, &State::set_label)
    .def_property_readonly("is_cuda", &State::is_cuda)
    .def_property_readonly("device", &State::get_device)
    .def_property_readonly("dtype", &State::get_dtype)
    .def_property_readonly("numel", &State::get_numel)
    .def_property_readonly("is_tt", &State::is_tt);

  // =================================================================
  // Core Methods (Release GIL: Heavy compute / memory transfers)
  state
    .def(
      "defined", &State::defined, "Check if the underlying data is allocated")
    .def("as_tt", &State::as_tt, py::return_value_policy::reference_internal)

    .def("to_buffer", &State::to_buffer, py::arg("buffer"),
      py::call_guard<py::gil_scoped_release>())
    .def("from_buffer", &State::from_buffer, py::arg("buffer"),
      py::call_guard<py::gil_scoped_release>())
    .def("neg_", &State::neg_, py::call_guard<py::gil_scoped_release>())

    .def("round_", &State::round_, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())
    .def("round", &State::round, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // Device & Dtype Transfers (to / to_)

  // In-place to_()
  state
    .def("to_",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(&State::to_),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&State::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&State::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&State::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  // Out-of-place to()
  state
    .def("to",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &State::to, py::const_),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&State::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&State::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&State::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // Python Magic Methods for Math (Release GIL)

  // Unary Negation
  state.def(
    "__neg__", [](const State& a) { return -a; },
    py::call_guard<py::gil_scoped_release>());

  // Addition
  state
    .def(
      "__add__", [](const State& a, const State& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__", [](const State& a, double b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__", [](const State& a, const torch::Tensor& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__", [](const State& a, double b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__", [](const State& a, const torch::Tensor& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>());

  // Subtraction
  state
    .def(
      "__sub__", [](const State& a, const State& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__", [](const State& a, double b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__", [](const State& a, const torch::Tensor& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__", [](const State& a, double b) { return b - a; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__", [](const State& a, const torch::Tensor& b) { return b - a; },
      py::call_guard<py::gil_scoped_release>());

  // Multiplication
  state
    .def(
      "__mul__", [](const State& a, const State& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__mul__", [](const State& a, double b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__mul__", [](const State& a, const torch::Tensor& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rmul__", [](const State& a, double b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rmul__", [](const State& a, const torch::Tensor& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>());

  // Division
  state
    .def(
      "__truediv__", [](const State& a, const State& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__truediv__", [](const State& a, double b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__truediv__",
      [](const State& a, const torch::Tensor& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rtruediv__", [](const State& a, double b) { return b / a; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rtruediv__",
      [](const State& a, const torch::Tensor& b) { return b / a; },
      py::call_guard<py::gil_scoped_release>());

  // In-place Operations
  state
    .def(
      "__iadd__",
      [](State& a, const State& b) -> State& {
        a += b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__iadd__",
      [](State& a, double b) -> State& {
        a += b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__iadd__",
      [](State& a, const torch::Tensor& b) -> State& {
        a += b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())

    .def(
      "__isub__",
      [](State& a, const State& b) -> State& {
        a -= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__isub__",
      [](State& a, double b) -> State& {
        a -= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__isub__",
      [](State& a, const torch::Tensor& b) -> State& {
        a -= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())

    .def(
      "__imul__",
      [](State& a, const State& b) -> State& {
        a *= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__imul__",
      [](State& a, double b) -> State& {
        a *= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__imul__",
      [](State& a, const torch::Tensor& b) -> State& {
        a *= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())

    .def(
      "__itruediv__",
      [](State& a, const State& b) -> State& {
        a /= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__itruediv__",
      [](State& a, double b) -> State& {
        a /= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__itruediv__",
      [](State& a, const torch::Tensor& b) -> State& {
        a /= b;
        return a;
      },
      py::call_guard<py::gil_scoped_release>());
}
