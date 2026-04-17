#include "ttnte/linalg/tt_state.hpp"
#include <string>
#include <torch/extension.h>

namespace py = pybind11;

void register_TTState(py::module_& m)
{
  using namespace ttnte::linalg;

  py::class_<TTState, State, std::shared_ptr<TTState>>(m, "TTState")
    // =================================================================
    // Public constructors
    .def(py::init([](const std::vector<torch::Tensor>& cores,
                    std::optional<std::string> label) {
      return TTState::create(
        TTEngine::Tensors(cores.cbegin(), cores.cend()), label);
    }),
      py::arg("cores"), py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def_static(
      "clone_from",
      [](const std::vector<torch::Tensor>& cores,
        std::optional<std::string> label = std::nullopt) {
        return TTState::clone_from(
          TTEngine::Tensors(cores.cbegin(), cores.cend()), label);
      },
      py::arg("cores"), py::arg("label") = py::none())
    .def_static(
      "clone_from",
      [](const std::vector<torch::Tensor>& cores, const TTState::Label& label) {
        return TTState::clone_from(
          TTEngine::Tensors(cores.cbegin(), cores.cend()), label);
      },
      py::arg("cores"), py::arg("label"))
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("get_engine", &TTState::get_engine,
      py::return_value_policy::reference_internal)
    .def("get_cores", &TTState::get_cores,
      py::return_value_policy::reference_internal)

    // =================================================================
    // Public getters / setters
    .def_property_readonly("engine", &TTState::get_engine)
    .def_property_readonly("cores",
      [](const TTState& self) {
        const auto& cores = self.get_cores();
        return std::vector<torch::Tensor>(cores.begin(), cores.end());
      })
    .def_property_readonly("device", &TTState::get_device)
    .def_property_readonly("dtype", &TTState::get_dtype)
    .def_property_readonly("numel", &TTState::get_numel);
}
