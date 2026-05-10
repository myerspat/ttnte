#include "ttnte/linalg/state.hpp"
#include "../utils/label.hpp"
#include <torch/extension.h>

namespace py = pybind11;

class PyState : public ttnte::linalg::State {
public:
  using State::State; // Inherit constructors

  // Trampoline for pure virtual to_impl
  Ptr to_impl(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking, bool copy,
    std::optional<at::MemoryFormat> memory_format) const override
  {
    PYBIND11_OVERRIDE_PURE(
      Ptr, State, to_impl, device, dtype, non_blocking, copy, memory_format);
  }

  // Trampolines for pure virtual to_ methods
  void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) override
  {
    PYBIND11_OVERRIDE_PURE(
      void, State, to_, device, dtype, non_blocking, copy, memory_format);
  }

  void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) override
  {
    PYBIND11_OVERRIDE_PURE(
      void, State, to_, dtype, non_blocking, copy, memory_format);
  }

  void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) override
  {
    PYBIND11_OVERRIDE_PURE(
      void, State, to_, device, non_blocking, copy, memory_format);
  }

  // Trampoline for virtual pack method
  torch::Tensor pack() const override
  {
    PYBIND11_OVERRIDE(torch::Tensor, State, pack);
  }

  // Trampoline for pure virtual get_device
  torch::Device get_device() const override
  {
    PYBIND11_OVERRIDE_PURE(torch::Device, State, get_device);
  }

  // Trampoline for pure virtual get_dtype
  at::ScalarType get_dtype() const override
  {
    PYBIND11_OVERRIDE_PURE(at::ScalarType, State, get_dtype);
  }

  // Trampoline for pure virtual get_numel
  int64_t get_numel() const override
  {
    PYBIND11_OVERRIDE_PURE(int64_t, State, get_numel);
  }
};

void register_State(py::module_& m)
{
  using namespace ttnte::linalg;
  register_Label<State>(m, "State");

  py::class_<State, PyState, State::Ptr>(m, "State")
    // =================================================================
    // Public methods
    .def("is_cuda", &State::is_cuda)

    // Send/Cast operations can be heavy; release the GIL
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
      py::call_guard<py::gil_scoped_release>())

    // In-place Send/Cast operations; release the GIL
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
      py::call_guard<py::gil_scoped_release>())

    // Pack and unpack involve buffer copies/allocations; release the GIL
    .def("pack", &State::pack, py::call_guard<py::gil_scoped_release>())
    .def_static("unpack", &State::unpack, py::arg("buffer"),
      py::arg("clone") = true, py::call_guard<py::gil_scoped_release>())

    .def("get_label", &State::get_label,
      py::return_value_policy::reference_internal)

    // =================================================================
    // Public getters / setters
    .def_property_readonly("label", &State::get_label)
    .def_property_readonly("device", &State::get_device)
    .def_property_readonly("dtype", &State::get_dtype)
    .def_property_readonly("numel", &State::get_numel);
}
