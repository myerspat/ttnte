#include "ttnte/linalg/linear_system.hpp"
#include "../utils/label.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_LinearSystem(py::module_& m)
{
  using namespace ttnte::linalg;

  register_Label<LinearSystem>(m, "LinearSystem");
  py::class_<LinearSystem, std::shared_ptr<LinearSystem>> ls(
    m, "LinearSystem", "ttnte abstract linear system");

  // =================================================================
  // Constructors (Factory pattern using lambda)
  ls.def(py::init([](Operator interior_op, State state, State source,
                    std::optional<std::string> label) {
    return LinearSystem::create(std::move(interior_op), std::move(state),
      std::move(source), std::move(label));
  }),
    py::arg("interior_op"), py::arg("state") = State(),
    py::arg("source") = State(), py::arg("label") = py::none(),
    "Constructs a LinearSystem via the create factory method");

  // =================================================================
  // Properties & Getters/Setters (Keep GIL: Trivial accesses)
  ls.def_property_readonly(
      "label", [](const LinearSystem& l) { return l.get_label().to_string(); })
    .def_property_readonly("interior_op", &LinearSystem::get_interior_op)
    .def_property("state", &LinearSystem::get_state, &LinearSystem::set_state)
    .def_property(
      "source", &LinearSystem::get_source, &LinearSystem::set_source)
    .def_property_readonly("dtype", &LinearSystem::get_dtype);

  ls.def("get_buffer", &LinearSystem::get_buffer, py::arg("device"),
    py::return_value_policy::reference_internal,
    "Returns the underlying buffer tensor for a given device.");

  // =================================================================
  // Device & Dtype Transfers (Release GIL: High-latency memory IO)

  // transfer_buffer overloads
  ls.def("transfer_buffer",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &LinearSystem::transfer_buffer),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("transfer_buffer",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_buffer),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("transfer_buffer",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_buffer),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  // transfer_nonbuffer overloads
  ls.def("transfer_nonbuffer",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &LinearSystem::transfer_nonbuffer),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("transfer_nonbuffer",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_nonbuffer),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("transfer_nonbuffer",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_nonbuffer),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  // to_ overloads (In-place linear system cast)
  ls.def("to_",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &LinearSystem::to_),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());
}
