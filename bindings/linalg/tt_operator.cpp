#include "ttnte/linalg/tt_operator.hpp"
#include <string>
#include <torch/extension.h>
#include <vector>

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
    // Factory Methods
    .def_static(
      "zeros",
      [](const std::vector<int64_t>& m_modes,
        const std::vector<int64_t>& n_modes,
        std::optional<torch::Device> device,
        std::optional<torch::ScalarType> dtype) {
        return TTOperator::zeros(
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()),
          c10::SmallVector<int64_t, 6>(n_modes.begin(), n_modes.end()), device,
          dtype);
      },
      py::arg("m_modes"), py::arg("n_modes"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none())
    .def_static(
      "ones",
      [](const std::vector<int64_t>& m_modes,
        const std::vector<int64_t>& n_modes,
        std::optional<torch::Device> device,
        std::optional<torch::ScalarType> dtype) {
        return TTOperator::ones(
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()),
          c10::SmallVector<int64_t, 6>(n_modes.begin(), n_modes.end()), device,
          dtype);
      },
      py::arg("m_modes"), py::arg("n_modes"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none())
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
    .def_static(
      "from_dense",
      [](torch::Tensor tensor, const std::vector<int64_t>& m_modes,
        const std::vector<int64_t>& n_modes, double eps, int64_t max_rank,
        bool is_interleaved, std::optional<std::string> label) {
        return TTOperator::from_dense(tensor,
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()),
          c10::SmallVector<int64_t, 6>(n_modes.begin(), n_modes.end()), eps,
          max_rank, is_interleaved, label);
      },
      py::arg("tensor"), py::arg("m_modes"), py::arg("n_modes"),
      py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::arg("is_interleaved") = false, py::arg("label") = py::none(),
      py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Transpose tensors
    .def("transpose_", [](TTOperator::Ptr& self, const std::vector<int64_t>& core_idxs = {}) { self->transpose_(c10::SmallVector<int64_t, 6>(core_idxs.cbegin(), core_idxs.cend()));}, py::arg("core_idxs") = std::vector<int64_t>{})
    .def("transpose", [](const TTOperator::Ptr& self, const std::vector<int64_t>& core_idxs = {}) {return self->transpose(c10::SmallVector<int64_t, 6>(core_idxs.cbegin(), core_idxs.cend()));}, py::arg("core_idxs") = std::vector<int64_t>{})

    // =================================================================
    // Casting and Device Transfers (with GIL release)
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())

    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTOperator::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Tensor Train Operations (with GIL release)
    .def("lr_orthogonalize_", &TTOperator::lr_orthogonalize_,
      py::call_guard<py::gil_scoped_release>())
    .def("lr_orthogonalize", &TTOperator::lr_orthogonalize,
      py::call_guard<py::gil_scoped_release>())

    .def("round_", &TTOperator::round_, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())
    .def("round", &TTOperator::round, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())

    .def("to_dense", &TTOperator::to_dense, py::arg("interleave") = false,
      py::call_guard<py::gil_scoped_release>())

    .def("from_buffer", &TTOperator::from_buffer, py::arg("buffer"),
      py::call_guard<py::gil_scoped_release>())

    .def("neg_", &TTOperator::neg_, py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Mathematical Operators
    // Addition
    .def(
      "__add__",
      [](const TTOperator::Ptr& a, const TTOperator::Ptr& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__", [](const TTOperator::Ptr& a, double b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__",
      [](const TTOperator::Ptr& a, const torch::Tensor& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__", [](const TTOperator::Ptr& b, double a) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__",
      [](const TTOperator::Ptr& b, const torch::Tensor& a) { return a + b; },
      py::call_guard<py::gil_scoped_release>())

    // Subtraction
    .def(
      "__sub__",
      [](const TTOperator::Ptr& a, const TTOperator::Ptr& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__", [](const TTOperator::Ptr& a, double b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__",
      [](const TTOperator::Ptr& a, const torch::Tensor& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__", [](const TTOperator::Ptr& b, double a) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__",
      [](const TTOperator::Ptr& b, const torch::Tensor& a) { return a - b; },
      py::call_guard<py::gil_scoped_release>())

    // Multiplication
    .def(
      "__mul__",
      [](const TTOperator::Ptr& a, const TTOperator::Ptr& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__mul__", [](const TTOperator::Ptr& a, double b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__mul__",
      [](const TTOperator::Ptr& a, const torch::Tensor& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rmul__", [](const TTOperator::Ptr& b, double a) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rmul__",
      [](const TTOperator::Ptr& b, const torch::Tensor& a) { return a * b; },
      py::call_guard<py::gil_scoped_release>())

    // Division
    .def(
      "__truediv__",
      [](const TTOperator::Ptr& a, const TTOperator::Ptr& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__truediv__", [](const TTOperator::Ptr& a, double b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__truediv__",
      [](const TTOperator::Ptr& a, const torch::Tensor& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rtruediv__", [](const TTOperator::Ptr& b, double a) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rtruediv__",
      [](const TTOperator::Ptr& b, const torch::Tensor& a) { return a / b; },
      py::call_guard<py::gil_scoped_release>())

    // Iterators and length
    .def("__len__", &TTEngine::size)
    .def("__getitem__", [](const TTOperator::Ptr& self, size_t i) {
      size_t len = self->size();

      // Handle negative indices
      if (i < 0) i += len;
      if (i < 0 || i >= len) {
          throw py::index_error("TTState index out of range");
      }
      return (*self)[i];
    })
    .def("__iter__", [](const TTEngine& self) {
      return py::make_iterator(self.begin(), self.end());
    }, py::keep_alive<0, 1>())

    // =================================================================
    // Public getters / Properties
    .def("get_engine", &TTOperator::get_engine,
      py::return_value_policy::reference_internal)
    .def("get_cores", &TTOperator::get_cores,
      py::return_value_policy::reference_internal)

    .def_property_readonly("engine", &TTOperator::get_engine)
    .def_property_readonly("cores",
      [](const TTOperator& self) {
        const auto& cores = self.get_cores();
        return std::vector<torch::Tensor>(cores.begin(), cores.end());
      })
    .def_property_readonly("device", &TTOperator::get_device)
    .def_property_readonly("dtype", &TTOperator::get_dtype)
    .def_property_readonly("numel", &TTOperator::get_numel)
    .def_property_readonly("ranks",
      [](const TTOperator& self) {
        auto r = self.get_ranks();
        return std::vector<int64_t>(r.begin(), r.end());
      })
    .def_property_readonly("free_indices",
      [](const TTOperator& self) {
        auto f = self.get_free_indices();
        return std::vector<int64_t>(f.begin(), f.end());
      })
    .def_property_readonly("m_modes",
      [](const TTOperator& self) {
        auto m = self.get_m_modes();
        return std::vector<int64_t>(m.begin(), m.end());
      })
    .def_property_readonly("n_modes", [](const TTOperator& self) {
      auto n = self.get_n_modes();
      return std::vector<int64_t>(n.begin(), n.end());
    });
}
