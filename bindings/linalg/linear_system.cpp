#include "ttnte/linalg/linear_system.hpp"
#include "../utils/label.hpp"
#include <torch/extension.h>

namespace py = pybind11;

class PyLinearSystem : public ttnte::linalg::LinearSystem {
public:
  using LinearSystem::LinearSystem;

  // Trampolines for to_ overloads
  void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking, bool copy,
    std::optional<at::MemoryFormat> memory_format) override
  {
    PYBIND11_OVERRIDE_PURE(void, LinearSystem, to_, device, dtype, non_blocking,
      copy, memory_format);
  }
  void to_(const torch::Device& device, bool non_blocking, bool copy,
    std::optional<at::MemoryFormat> memory_format) override
  {
    PYBIND11_OVERRIDE_PURE(
      void, LinearSystem, to_, device, non_blocking, copy, memory_format);
  }
  void to_(const at::ScalarType& dtype, bool non_blocking, bool copy,
    std::optional<at::MemoryFormat> memory_format) override
  {
    PYBIND11_OVERRIDE_PURE(
      void, LinearSystem, to_, dtype, non_blocking, copy, memory_format);
  }

  // Trampolines for transfer_nonbuffer overloads
  void transfer_nonbuffer(const torch::Device& device,
    const at::ScalarType& dtype, bool non_blocking, bool copy,
    std::optional<at::MemoryFormat> memory_format) override
  {
    PYBIND11_OVERRIDE_PURE(void, LinearSystem, transfer_nonbuffer, device,
      dtype, non_blocking, copy, memory_format);
  }
  void transfer_nonbuffer(const torch::Device& device, bool non_blocking,
    bool copy, std::optional<at::MemoryFormat> memory_format) override
  {
    PYBIND11_OVERRIDE_PURE(void, LinearSystem, transfer_nonbuffer, device,
      non_blocking, copy, memory_format);
  }

  // Trampolines for Getters
  OpPtr get_interior_op() const override
  {
    PYBIND11_OVERRIDE_PURE(OpPtr, LinearSystem, get_interior_op);
  }
  StPtr get_state() const override
  {
    PYBIND11_OVERRIDE_PURE(StPtr, LinearSystem, get_state);
  }
  StPtr get_source() const override
  {
    PYBIND11_OVERRIDE_PURE(StPtr, LinearSystem, get_source);
  }
};

void register_LinearSystem(py::module_& m)
{
  using namespace ttnte::linalg;

  py::class_<LinearSystem, PyLinearSystem, std::shared_ptr<LinearSystem>>(
    m, "LinearSystem")
    // =================================================================
    // Public methods
    .def("is_finalized", &LinearSystem::is_finalized)

    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("transfer_buffer",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_buffer),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("transfer_buffer",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_buffer),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("transfer_nonbuffer",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_nonbuffer),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("transfer_nonbuffer",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_nonbuffer),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("get_label", &LinearSystem::get_label,
      py::return_value_policy::reference_internal)
    .def("get_buffer", &LinearSystem::get_buffer, py::arg("device"),
      py::return_value_policy::reference_internal)

    // =================================================================
    // Public getters / setters
    .def_property_readonly("label", &LinearSystem::get_label)
    .def_property_readonly("interior_op", &LinearSystem::get_interior_op)
    .def_property_readonly("state", &LinearSystem::get_state)
    .def_property_readonly("source", &LinearSystem::get_source)
    .def_property_readonly("dtype", &LinearSystem::get_dtype);
}
