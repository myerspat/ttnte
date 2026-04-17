#include "ttnte/linalg/tt_engine.hpp"
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

void register_TTEngine(py::module_& m)
{
  using namespace ttnte::linalg;

  py::class_<TTEngine>(m, "TTEngine")
    // =================================================================
    // Public constructors
    .def(py::init([](const std::vector<torch::Tensor>& cores,
                    bool check_cores = true) {
      return TTEngine(TTEngine::Tensors(cores.cbegin(), cores.cend()));
    }),
      py::arg("cores"), py::arg("check_cores") = true)

    // =================================================================
    // Public methods
    .def_static(
      "clone_from",
      [](const std::vector<torch::Tensor>& cores, bool check_cores = true) {
        return TTEngine::clone_from(
          TTEngine::Tensors(cores.cbegin(), cores.cend()), check_cores);
      },
      py::arg("cores"), py::arg("check_cores") = true)
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::return_value_policy::reference_internal)
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::return_value_policy::reference_internal)
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::return_value_policy::reference_internal)

    .def("from_buffer", &TTEngine::from_buffer, py::arg("buffer"))

    // =================================================================
    // Public getters / setters
    .def_property_readonly("cores",
      [](const TTEngine& self) {
        const auto& cores = self.get_cores();
        return std::vector<torch::Tensor>(cores.begin(), cores.end());
      })
    .def_property_readonly("device", &TTEngine::get_device)
    .def_property_readonly("dtype", &TTEngine::get_dtype)
    .def_property_readonly("numel", &TTEngine::get_numel)
    .def_property_readonly("ranks",
      [](const TTEngine& self) {
        const auto& ranks = self.get_ranks();
        return std::vector<int64_t>(ranks.begin(), ranks.end());
      })
    .def_property_readonly("free_indices", [](const TTEngine& self) {
      const auto& free_indices = self.get_free_indices();
      return std::vector<int64_t>(free_indices.begin(), free_indices.end());
    });
}
