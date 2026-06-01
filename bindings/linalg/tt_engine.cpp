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
      return TTEngine(
        TTEngine::Tensors(cores.cbegin(), cores.cend()), check_cores);
    }),
      py::arg("cores"), py::arg("check_cores") = true)

    // =================================================================
    // Factory methods
    .def_static(
      "clone_from",
      [](const std::vector<torch::Tensor>& cores, bool check_cores = true) {
        return TTEngine::clone_from(
          TTEngine::Tensors(cores.cbegin(), cores.cend()), check_cores);
      },
      py::arg("cores"), py::arg("check_cores") = true)

    // TT-SVD
    .def_static(
      "tt_svd",
      [](torch::Tensor tensor, double eps, int64_t max_rank) {
        auto cores = TTEngine::tt_svd(tensor, eps, max_rank);
        return std::vector<torch::Tensor>(cores.begin(), cores.end());
      },
      py::arg("tensor"), py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())

    // from_dense (vector overload)
    .def_static("from_dense",
      py::overload_cast<const torch::Tensor&, double, int64_t>(
        &TTEngine::from_dense),
      py::arg("tensor"), py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())

    // from_dense (matrix overload)
    .def_static(
      "from_dense",
      [](torch::Tensor tensor, const std::vector<int64_t>& m_modes,
        const std::vector<int64_t>& n_modes, double eps, int64_t max_rank,
        bool is_interleaved) {
        return TTEngine::from_dense(tensor,
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()),
          c10::SmallVector<int64_t, 6>(n_modes.begin(), n_modes.end()), eps,
          max_rank, is_interleaved);
      },
      py::arg("tensor"), py::arg("m_modes"), py::arg("n_modes"),
      py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::arg("is_interleaved") = false,
      py::call_guard<py::gil_scoped_release>())

    // Zeros
    .def_static(
      "zeros",
      [](const std::vector<int64_t>& m_modes,
        std::optional<torch::Device> device,
        std::optional<torch::ScalarType> dtype) {
        return TTEngine::zeros(
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()), device,
          dtype);
      },
      py::arg("m_modes"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none())
    .def_static(
      "zeros",
      [](const std::vector<int64_t>& m_modes,
        const std::vector<int64_t>& n_modes,
        std::optional<torch::Device> device,
        std::optional<torch::ScalarType> dtype) {
        return TTEngine::zeros(
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()),
          c10::SmallVector<int64_t, 6>(n_modes.begin(), n_modes.end()), device,
          dtype);
      },
      py::arg("m_modes"), py::arg("n_modes"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none())

    // Ones
    .def_static(
      "ones",
      [](const std::vector<int64_t>& m_modes,
        std::optional<torch::Device> device,
        std::optional<torch::ScalarType> dtype) {
        return TTEngine::ones(
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()), device,
          dtype);
      },
      py::arg("m_modes"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none())
    .def_static(
      "ones",
      [](const std::vector<int64_t>& m_modes,
        const std::vector<int64_t>& n_modes,
        std::optional<torch::Device> device,
        std::optional<torch::ScalarType> dtype) {
        return TTEngine::ones(
          c10::SmallVector<int64_t, 6>(m_modes.begin(), m_modes.end()),
          c10::SmallVector<int64_t, 6>(n_modes.begin(), n_modes.end()), device,
          dtype);
      },
      py::arg("m_modes"), py::arg("n_modes"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none())

    // Meshgrid
    .def_static("meshgrid", [](const std::vector<torch::Tensor>& vecs) {
        auto result = TTEngine::meshgrid(TTEngine::Tensors(vecs.cbegin(), vecs.cend()));
        return std::vector<TTEngine>(result.begin(), result.end());
    }, py::arg("vecs"), py::call_guard<py::gil_scoped_release>())


    // =================================================================
    // Transpose tensors
    .def("transpose_",
      [](TTEngine& self, const std::vector<int64_t>& core_idxs = {}) {
        return self.transpose_(c10::SmallVector<int64_t, 6>(core_idxs.cbegin(), core_idxs.cend()));
      },
      py::arg("core_idxs") = std::vector<int64_t>{},
      py::return_value_policy::reference_internal)
    .def("transpose",
      [](const TTEngine& self, const std::vector<int64_t>& core_idxs = {}) {
        return self.transpose(c10::SmallVector<int64_t, 6>(core_idxs.cbegin(), core_idxs.cend()));
      },
      py::arg("core_idxs") = std::vector<int64_t>{})

    // =================================================================
    // Casting and Device Transfers (with GIL release)
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())

    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&TTEngine::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::TensorOptions&>(&TTEngine::to_),
      py::arg("options"),
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Tensor Train Operations (with GIL release)
    .def("lr_orthogonalize_", &TTEngine::lr_orthogonalize_,
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("lr_orthogonalize", &TTEngine::lr_orthogonalize,
      py::call_guard<py::gil_scoped_release>())

    .def("round_", &TTEngine::round_, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("round", &TTEngine::round, py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())

    .def("to_dense", &TTEngine::to_dense,
        py::arg("interleave") = false,
      py::call_guard<py::gil_scoped_release>())

    .def("from_buffer", &TTEngine::from_buffer, py::arg("buffer"),
      py::call_guard<py::gil_scoped_release>())

    .def("neg_", &TTEngine::neg_, py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())

    .def("norm", &TTEngine::norm, py::call_guard<py::gil_scoped_release>())
    .def("sum", &TTEngine::sum, py::call_guard<py::gil_scoped_release>())

    .def("diagonalize_",
      [](TTEngine& self, const std::vector<int64_t>& core_idxs = {}) {
        return self.diagonalize_(c10::SmallVector<int64_t, 6>(core_idxs.cbegin(), core_idxs.cend()));
      },
      py::arg("core_idxs") = std::vector<int64_t>{},
      py::return_value_policy::reference_internal)
    .def("diagonalize",
      [](const TTEngine& self, const std::vector<int64_t>& core_idxs = {}) {
        return self.diagonalize(c10::SmallVector<int64_t, 6>(core_idxs.cbegin(), core_idxs.cend()));
      },
      py::arg("core_idxs") = std::vector<int64_t>{})

    .def("is_rank_one", &TTEngine::is_rank_one)

    .def("expand_",
      [](TTEngine& self, const std::vector<int64_t>& m_modes, const std::vector<int64_t>& n_modes){
        return self.expand_(
            c10::SmallVector<int64_t, 6>(m_modes.cbegin(), m_modes.cend()),
            c10::SmallVector<int64_t, 6>(n_modes.cbegin(), n_modes.cend()));
      },
      py::arg("m_modes"), py::arg("n_modes"),
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("expand",
      [](const TTEngine& self, const std::vector<int64_t>& m_modes, const std::vector<int64_t>& n_modes){
        return self.expand(
            c10::SmallVector<int64_t, 6>(m_modes.cbegin(), m_modes.cend()),
            c10::SmallVector<int64_t, 6>(n_modes.cbegin(), n_modes.cend()));
      },
      py::arg("m_modes"), py::arg("n_modes"),
      py::call_guard<py::gil_scoped_release>())

    .def("contract_rank_dim_", &TTEngine::contract_rank_dim_,
             py::arg("dim"), py::return_value_policy::reference_internal,
             py::call_guard<py::gil_scoped_release>())
    .def("contract_rank_dim", &TTEngine::contract_rank_dim,
          py::arg("dim"), py::call_guard<py::gil_scoped_release>())

    .def("kron_", py::overload_cast<const TTEngine&>(&TTEngine::kron_),
          py::arg("other"), py::return_value_policy::reference_internal)
    .def("kron", py::overload_cast<const TTEngine&>(&TTEngine::kron, py::const_),
          py::arg("other"))
    .def("kron_", py::overload_cast<const torch::Tensor&>(&TTEngine::kron_),
          py::arg("other"), py::return_value_policy::reference_internal)
    .def("kron", py::overload_cast<const torch::Tensor&>(&TTEngine::kron, py::const_),
          py::arg("other"))

    .def("evaluate_at", &TTEngine::evaluate_at, py::arg("indices"),
        py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Mathematical Operators
    .def(
      "__neg__", &TTEngine::operator-, py::call_guard<py::gil_scoped_release>())
    // Addition
    .def(
      "__add__", [](const TTEngine& a, const TTEngine& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__", [](const TTEngine& a, double b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__add__",
      [](const TTEngine& a, const torch::Tensor& b) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__", [](const TTEngine& b, double a) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__radd__",
      [](const TTEngine& b, const torch::Tensor& a) { return a + b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__iadd__",
      [](TTEngine& a, const TTEngine& b) -> TTEngine& { return a += b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__iadd__", [](TTEngine& a, double b) -> TTEngine& { return a += b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__iadd__",
      [](TTEngine& a, const torch::Tensor& b) -> TTEngine& { return a += b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())

    // Subtraction
    .def(
      "__sub__", [](const TTEngine& a, const TTEngine& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__", [](const TTEngine& a, double b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__sub__",
      [](const TTEngine& a, const torch::Tensor& b) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__", [](const TTEngine& b, double a) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__rsub__",
      [](const TTEngine& b, const torch::Tensor& a) { return a - b; },
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__isub__",
      [](TTEngine& a, const TTEngine& b) -> TTEngine& { return a -= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__isub__", [](TTEngine& a, double b) -> TTEngine& { return a -= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def(
      "__isub__",
      [](TTEngine& a, const torch::Tensor& b) -> TTEngine& { return a -= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())

    // Multiplication
    .def("__mul__",
      [](const TTEngine& a, const TTEngine& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__mul__",
      [](const TTEngine& a, double b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__mul__",
      [](const TTEngine& a, const torch::Tensor& b) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__rmul__",
      [](const TTEngine& b, double a) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__rmul__",
      [](const TTEngine& b, const torch::Tensor& a) { return a * b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__imul__",
      [](TTEngine& a, const TTEngine& b) -> TTEngine& { return a *= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("__imul__",
      [](TTEngine& a, double b) -> TTEngine& { return a *= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("__imul__",
      [](TTEngine& a, const torch::Tensor& b) -> TTEngine& { return a *= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())

    // Division
    .def("__truediv__",
      [](const TTEngine& a, const TTEngine& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__truediv__",
      [](const TTEngine& a, double b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__truediv__",
      [](const TTEngine& a, const torch::Tensor& b) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__rtruediv__",
      [](const TTEngine& b, double a) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__rtruediv__",
      [](const TTEngine& b, const torch::Tensor& a) { return a / b; },
      py::call_guard<py::gil_scoped_release>())
    .def("__itruediv__",
      [](TTEngine& a, const TTEngine& b) -> TTEngine& { return a /= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("__itruediv__",
      [](TTEngine& a, double b) -> TTEngine& { return a /= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())
    .def("__itruediv__",
      [](TTEngine& a, const torch::Tensor& b) -> TTEngine& { return a /= b; },
      py::return_value_policy::reference_internal,
      py::call_guard<py::gil_scoped_release>())

    // Iterators and length
    .def("__len__", &TTEngine::size)
    .def("__getitem__", [](const TTEngine& self, size_t i) {
      size_t len = self.size();

      // Handle negative indices
      if (i < 0) i += len;
      if (i < 0 || i >= len) {
          throw py::index_error("TTState index out of range");
      }
      return self[i];
    })
    .def("__iter__", [](const TTEngine& self) {
      return py::make_iterator(self.begin(), self.end());
    }, py::keep_alive<0, 1>())


    // =================================================================
    // Public getters / setters
    .def_property_readonly("size", &TTEngine::size)
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
    .def_property_readonly("free_indices",
      [](const TTEngine& self) {
        const auto& free_indices = self.get_free_indices();
        return std::vector<int64_t>(free_indices.begin(), free_indices.end());
      })
    .def_property_readonly("m_modes",
      [](const TTEngine& self) {
        const auto& m_modes = self.get_m_modes();
        return std::vector<int64_t>(m_modes.begin(), m_modes.end());
      })
    .def_property_readonly("n_modes", [](const TTEngine& self) {
      const auto& n_modes = self.get_n_modes();
      return std::vector<int64_t>(n_modes.begin(), n_modes.end());
    });
}
