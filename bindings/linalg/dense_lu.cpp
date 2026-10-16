#include "ttnte/linalg/dense_lu.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_dense_lu(py::module_& m)
{
  using namespace ttnte::linalg;

  m.def(
    "dense_lu_factor",
    [](const torch::Tensor& a) {
      auto factored = a.to(torch::kFloat64).contiguous().clone();
      auto ipiv = dense::batched_lu_factor_(factored);
      return std::make_tuple(factored, ipiv);
    },
    "Factor a batch of (batch, n, n) matrices with the raw small-dense LU. "
    "Returns (factored matrices, pivots).",
    py::arg("a"), py::call_guard<py::gil_scoped_release>());

  m.def(
    "dense_lu_solve",
    [](const torch::Tensor& a, const torch::Tensor& ipiv,
      const torch::Tensor& b) {
      auto x = b.to(torch::kFloat64).contiguous().clone();
      dense::batched_lu_solve_(a, ipiv, x);
      return x;
    },
    "Solve factored systems for a batch of right-hand sides (batch, n).",
    py::arg("a"), py::arg("ipiv"), py::arg("b"),
    py::call_guard<py::gil_scoped_release>());
}
