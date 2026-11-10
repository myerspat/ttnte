#include "ttnte/linalg/banded_lu.hpp"
#include "ttnte/utils/exception.hpp"
#include <complex>
#include <torch/extension.h>

namespace py = pybind11;

namespace {

using ttnte::linalg::banded::factor_ldab;
using ttnte::linalg::banded::lu_factor;
using ttnte::linalg::banded::lu_solve;

std::complex<double>* zptr(torch::Tensor& t)
{
  return reinterpret_cast<std::complex<double>*>(
    t.data_ptr<c10::complex<double>>());
}

/// Complex batched factor (zgbtrf per matrix); the real path is
/// banded::batched_lu_factor_().
torch::Tensor complex_lu_factor_(torch::Tensor& ab, int64_t kl, int64_t ku)
{
  const char* where = "ttnte::linalg::banded_lu_factor";
  if (ab.dim() != 3 || ab.size(2) < factor_ldab(kl, ku))
    throw ttnte::utils::runtime_error(
      where, "Expected (batch, n, ldab) storage with ldab >= 2 kl + ku + 1");
  const int64_t batch = ab.size(0), n = ab.size(1), ldab = ab.size(2);
  auto ipiv = torch::empty({batch, n}, torch::kInt64);
  for (int64_t k = 0; k < batch; k++) {
    auto m = ab[k];
    if (lu_factor(zptr(m), ldab, n, kl, ku, ipiv[k].data_ptr<int64_t>()) != 0)
      throw ttnte::utils::runtime_error(where, "Singular band matrix");
  }
  return ipiv;
}

void complex_lu_solve_(const torch::Tensor& ab, const torch::Tensor& ipiv,
  int64_t kl, int64_t ku, torch::Tensor& b)
{
  const char* where = "ttnte::linalg::banded_lu_solve";
  if (ab.dim() != 3 || ab.size(2) < factor_ldab(kl, ku) || b.dim() != 2 ||
      b.size(0) != ab.size(0) || b.size(1) != ab.size(1))
    throw ttnte::utils::runtime_error(
      where, "Expected (batch, n, ldab) factors and (batch, n) right sides");
  auto a = ab.contiguous();
  auto piv = ipiv.to(torch::kInt64).contiguous();
  const int64_t batch = a.size(0), n = a.size(1), ldab = a.size(2);
  for (int64_t k = 0; k < batch; k++) {
    auto m = a[k];
    auto rhs = b[k];
    lu_solve(zptr(m), ldab, n, kl, ku, piv[k].data_ptr<int64_t>(), zptr(rhs));
  }
}

} // namespace

void register_banded_lu(py::module_& m)
{
  using namespace ttnte::linalg;

  m.def(
    "banded_lu_factor",
    [](const torch::Tensor& ab, int64_t kl, int64_t ku) {
      if (ab.is_complex()) {
        auto factored = ab.to(torch::kComplexDouble).contiguous().clone();
        auto ipiv = complex_lu_factor_(factored, kl, ku);
        return std::make_tuple(factored, ipiv);
      }
      auto factored = ab.to(torch::kFloat64).contiguous().clone();
      auto ipiv = banded::batched_lu_factor_(factored, kl, ku);
      return std::make_tuple(factored, ipiv);
    },
    "Factor a batch of band matrices (LAPACK band storage, shape (batch, n, "
    "2kl+ku+1)), real (dgbtrf) or complex (zgbtrf). Returns (factored band, "
    "pivots).",
    py::arg("ab"), py::arg("kl"), py::arg("ku"),
    py::call_guard<py::gil_scoped_release>());

  m.def(
    "banded_lu_solve",
    [](const torch::Tensor& ab, const torch::Tensor& ipiv, int64_t kl,
      int64_t ku, const torch::Tensor& b) {
      if (ab.is_complex()) {
        auto x = b.to(torch::kComplexDouble).contiguous().clone();
        complex_lu_solve_(ab, ipiv, kl, ku, x);
        return x;
      }
      auto x = b.to(torch::kFloat64).contiguous().clone();
      banded::batched_lu_solve_(ab, ipiv, kl, ku, x);
      return x;
    },
    "Solve factored band systems for a batch of right-hand sides (batch, n); "
    "complex factors take complex right-hand sides.",
    py::arg("ab"), py::arg("ipiv"), py::arg("kl"), py::arg("ku"), py::arg("b"),
    py::call_guard<py::gil_scoped_release>());

  m.def("dense_to_band", &banded::dense_to_band,
    "Pack (batch, n, n) dense matrices into band storage with fill-in rows.",
    py::arg("a"), py::arg("kl"), py::arg("ku"));

  m.def("bandwidth", &banded::bandwidth,
    "Return (kl, ku), the sub- and super-diagonal bandwidth of a matrix.",
    py::arg("a"));
}
