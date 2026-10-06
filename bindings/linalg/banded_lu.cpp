#include "ttnte/linalg/banded_lu.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_banded_lu(py::module_& m)
{
  using namespace ttnte::linalg;

  m.def(
    "banded_lu_factor",
    [](const torch::Tensor& ab, int64_t kl, int64_t ku) {
      auto factored = ab.to(torch::kFloat64).contiguous().clone();
      auto ipiv = banded::batched_lu_factor_(factored, kl, ku);
      return std::make_tuple(factored, ipiv);
    },
    "Factor a batch of band matrices (LAPACK band storage, shape (batch, n, "
    "2kl+ku+1)). Returns (factored band, pivots).",
    py::arg("ab"), py::arg("kl"), py::arg("ku"),
    py::call_guard<py::gil_scoped_release>());

  m.def(
    "banded_lu_solve",
    [](const torch::Tensor& ab, const torch::Tensor& ipiv, int64_t kl,
      int64_t ku, const torch::Tensor& b) {
      auto x = b.to(torch::kFloat64).contiguous().clone();
      banded::batched_lu_solve_(ab, ipiv, kl, ku, x);
      return x;
    },
    "Solve factored band systems for a batch of right-hand sides (batch, n).",
    py::arg("ab"), py::arg("ipiv"), py::arg("kl"), py::arg("ku"), py::arg("b"),
    py::call_guard<py::gil_scoped_release>());

  m.def("dense_to_band", &banded::dense_to_band,
    "Pack (batch, n, n) dense matrices into band storage with fill-in rows.",
    py::arg("a"), py::arg("kl"), py::arg("ku"));

  m.def("bandwidth", &banded::bandwidth,
    "Return (kl, ku), the sub- and super-diagonal bandwidth of a matrix.",
    py::arg("a"));
}
