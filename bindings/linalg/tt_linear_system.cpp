#include "ttnte/linalg/tt_linear_system.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_TTLinearSystem(py::module_& m)
{
  using namespace ttnte::linalg;

  py::class_<TTLinearSystem, LinearSystem, std::shared_ptr<TTLinearSystem>>(
    m, "TTLinearSystem")
    // =================================================================
    // Public constructors
    .def(py::init([](const TTLinearSystem::OpPtr& interior_op,
                    const TTLinearSystem::StPtr& state,
                    const TTLinearSystem::StPtr& source,
                    std::optional<std::string> label) {
      return TTLinearSystem::create(interior_op, state, source, label);
    }),
      py::arg("interior_op"), py::arg("state"), py::arg("source"),
      py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTLinearSystem::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTLinearSystem::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTLinearSystem::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("transfer_buffer",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTLinearSystem::transfer_buffer),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("transfer_buffer",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTLinearSystem::transfer_buffer),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("transfer_nonbuffer",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTLinearSystem::transfer_nonbuffer),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("transfer_nonbuffer",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTLinearSystem::transfer_nonbuffer),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    // =================================================================
    // Public getters / setters
    .def_property_readonly("interior_op", &TTLinearSystem::get_interior_op)
    .def_property_readonly("state", &TTLinearSystem::get_state)
    .def_property_readonly("source", &TTLinearSystem::get_source);
}
