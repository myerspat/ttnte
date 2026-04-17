#include "ttnte/linalg/tt_operator.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_TTOperator(py::module_& m)
{
  using namespace ttnte::linalg;

  py::class_<TTOperator, Operator, std::shared_ptr<TTOperator>>(m, "TTOperator")
    // =================================================================
    // Public constructors
    .def(py::init([](const std::vector<torch::Tensor>& cores,
                    std::optional<std::string> label) {
      return TTOperator::create(
        TTEngine::Tensors(cores.cbegin(), cores.cend()), label);
    }),
      py::arg("cores"), py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def_static(
      "clone_from",
      [](const std::vector<torch::Tensor>& cores,
        std::optional<std::string> label = std::nullopt) {
        return TTOperator::clone_from(
          TTEngine::Tensors(cores.cbegin(), cores.cend()), label);
      },
      py::arg("cores"), py::arg("label") = py::none())
    .def_static(
      "clone_from",
      [](const std::vector<torch::Tensor>& cores,
        const TTOperator::Label& label) {
        return TTOperator::clone_from(
          TTEngine::Tensors(cores.cbegin(), cores.cend()), label);
      },
      py::arg("cores"), py::arg("label"))
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("get_engine", &TTOperator::get_engine,
      py::return_value_policy::reference_internal)

    // =================================================================
    // Public getters / setters
    .def_property_readonly("engine", &TTOperator::get_engine)
    .def_property_readonly("cores",
      [](const TTOperator& self) {
        const auto& cores = self.get_cores();
        return std::vector<torch::Tensor>(cores.begin(), cores.end());
      })
    .def_property_readonly("device", &TTOperator::get_device)
    .def_property_readonly("dtype", &TTOperator::get_dtype)
    .def_property_readonly("numel", &TTOperator::get_numel);
}
