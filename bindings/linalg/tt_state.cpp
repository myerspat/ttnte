#include "ttnte/linalg/tt_state.hpp"
#include <string>
#include <torch/extension.h>
#include <vector>

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
    // Factory Methods
    .def_static(
      "zeros",
      [](const std::vector<int64_t>& m_modes,
        std::optional<torch::Device> device,
        std::optional<torch::ScalarType> dtype) {
        return TTState::zeros(
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()), device,
          dtype);
      },
      py::arg("m_modes"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none())
    .def_static(
      "ones",
      [](const std::vector<int64_t>& m_modes,
        std::optional<torch::Device> device,
        std::optional<torch::ScalarType> dtype) {
        return TTState::ones(
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()), device,
          dtype);
      },
      py::arg("m_modes"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none())
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
    .def_static(
      "from_dense",
      [](const torch::Tensor& tensor, double eps, int64_t max_rank,
        std::optional<std::string> label) {
        return TTState::from_dense(tensor, eps, max_rank, label);
      },
      py::arg("tensor"), py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::arg("label") = py::none(), py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Casting and Device Transfers (with GIL release)
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())

    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTState::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Tensor Train Operations (with GIL release)
    .def("lr_orthogonalize_", &TTState::lr_orthogonalize_,
      py::call_guard<py::gil_scoped_release>())
    .def("lr_orthogonalize", &TTState::lr_orthogonalize,
      py::call_guard<py::gil_scoped_release>())

    .def("round_", &TTState::round_, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())
    .def("round", &TTState::round, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())

    .def(
      "to_dense", &TTState::to_dense, py::call_guard<py::gil_scoped_release>())

    .def("pack", &TTState::pack, py::call_guard<py::gil_scoped_release>())
    .def_static("unpack", &TTState::unpack, py::arg("buffer"),
      py::arg("clone") = true, py::call_guard<py::gil_scoped_release>())

    .def("neg_", &TTState::neg_, py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Transpose tensors
    .def(
      "transpose",
      [](const TTState::Ptr& self, const std::vector<int64_t>& core_idxs = {}) {
        return self->transpose(
          c10::SmallVector<int64_t, 6>(core_idxs.cbegin(), core_idxs.cend()));
      },
      py::arg("core_idxs") = std::vector<int64_t> {})

    // =================================================================
    // Mathematical Operators
    // Addition
    .def(
      "__add__",
      [](const TTState::Ptr& a, const TTState::Ptr& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__", [](const TTState::Ptr& a, double b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__",
      [](const TTState::Ptr& a, const torch::Tensor& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__", [](const TTState::Ptr& b, double a) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__",
      [](const TTState::Ptr& b, const torch::Tensor& a) { return a + b; },
      py::call_guard<py::gil_scoped_release>())

    // Subtraction
    .def(
      "__sub__",
      [](const TTState::Ptr& a, const TTState::Ptr& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__", [](const TTState::Ptr& a, double b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__",
      [](const TTState::Ptr& a, const torch::Tensor& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__", [](const TTState::Ptr& b, double a) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__",
      [](const TTState::Ptr& b, const torch::Tensor& a) { return a - b; },
      py::call_guard<py::gil_scoped_release>())

    // Multiplication
    .def(
      "__mul__",
      [](const TTState::Ptr& a, const TTState::Ptr& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__mul__", [](const TTState::Ptr& a, double b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__mul__",
      [](const TTState::Ptr& a, const torch::Tensor& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rmul__", [](const TTState::Ptr& b, double a) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rmul__",
      [](const TTState::Ptr& b, const torch::Tensor& a) { return a * b; },
      py::call_guard<py::gil_scoped_release>())

    // Division
    .def(
      "__truediv__",
      [](const TTState::Ptr& a, const TTState::Ptr& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__truediv__", [](const TTState::Ptr& a, double b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__truediv__",
      [](const TTState::Ptr& a, const torch::Tensor& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rtruediv__", [](const TTState::Ptr& b, double a) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rtruediv__",
      [](const TTState::Ptr& b, const torch::Tensor& a) { return a / b; },
      py::call_guard<py::gil_scoped_release>())

    // Iterators and length
    .def("__len__", &TTState::size)
    .def("__getitem__", [](const TTState::Ptr& self, size_t i) {
      size_t len = self->size();

      // Handle negative indices
      if (i < 0) i += len;
      if (i < 0 || i >= len) {
          throw py::index_error("TTState index out of range");
      }
      return (*self)[i];
    })
    .def("__iter__", [](const TTState::Ptr& self) {
      return py::make_iterator(self->begin(), self->end());
    }, py::keep_alive<0, 1>())

    // =================================================================
    // Public getters / Properties
    .def("get_engine", &TTState::get_engine,
      py::return_value_policy::reference_internal)
    .def("get_cores", &TTState::get_cores,
      py::return_value_policy::reference_internal)

    .def_property_readonly("engine", &TTState::get_engine)
    .def_property_readonly("cores",
      [](const TTState& self) {
        const auto& cores = self.get_cores();
        return std::vector<torch::Tensor>(cores.begin(), cores.end());
      })
    .def_property_readonly("device", &TTState::get_device)
    .def_property_readonly("dtype", &TTState::get_dtype)
    .def_property_readonly("numel", &TTState::get_numel)
    .def_property_readonly("ranks",
      [](const TTState& self) {
        auto r = self.get_ranks();
        return std::vector<int64_t>(r.begin(), r.end());
      })
    .def_property_readonly("free_indices",
      [](const TTState& self) {
        auto f = self.get_free_indices();
        return std::vector<int64_t>(f.begin(), f.end());
      })
    .def_property_readonly("m_modes", [](const TTState& self) {
      auto m = self.get_m_modes();
      return std::vector<int64_t>(m.begin(), m.end());
    });
}
