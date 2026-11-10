#include "ttnte/linalg/banded_cholesky.hpp"
#include "ttnte/utils/exception.hpp"
#include <torch/extension.h>

namespace py = pybind11;

namespace {

/// (n, kd + 1) float64 lower band storage, validated.
torch::Tensor checked_band(
  const torch::Tensor& ab, int64_t kd, const char* where)
{
  if (ab.dim() != 2 || ab.size(1) != kd + 1)
    throw ttnte::utils::runtime_error(
      where, "Expected (n, kd + 1) band storage");
  return ab.to(torch::kFloat64).contiguous().clone();
}

/// (n,) float64 vector matching the band, copied.
torch::Tensor checked_vector(
  const torch::Tensor& x, int64_t n, const char* where)
{
  if (x.dim() != 1 || x.size(0) != n)
    throw ttnte::utils::runtime_error(where, "Expected an (n,) vector");
  return x.to(torch::kFloat64).contiguous().clone();
}

} // namespace

void register_banded_cholesky(py::module_& m)
{
  using namespace ttnte::linalg;

  m.def(
    "banded_cholesky_factor",
    [](const torch::Tensor& ab, int64_t kd) {
      const char* where = "ttnte::linalg::banded_cholesky_factor";
      auto factor = checked_band(ab, kd, where);
      if (banded::cholesky_factor(
            factor.data_ptr<double>(), factor.size(0), kd) != 0)
        throw ttnte::utils::runtime_error(
          where, "The matrix is not positive definite");
      return factor;
    },
    "Cholesky factor A = L L^T of a symmetric positive definite band matrix "
    "in lower band storage ((n, kd + 1), row j = column j of the band). "
    "Returns L in the same storage.",
    py::arg("ab"), py::arg("kd"));

  m.def(
    "banded_cholesky_apply_upper",
    [](const torch::Tensor& factor, int64_t kd, const torch::Tensor& x) {
      const char* where = "ttnte::linalg::banded_cholesky_apply_upper";
      auto l = checked_band(factor, kd, where);
      auto y = checked_vector(x, l.size(0), where);
      banded::cholesky_apply_upper(
        l.data_ptr<double>(), l.size(0), kd, y.data_ptr<double>());
      return y;
    },
    "L^T x with the factor of banded_cholesky_factor().", py::arg("factor"),
    py::arg("kd"), py::arg("x"));

  m.def(
    "banded_cholesky_solve_upper",
    [](const torch::Tensor& factor, int64_t kd, const torch::Tensor& b) {
      const char* where = "ttnte::linalg::banded_cholesky_solve_upper";
      auto l = checked_band(factor, kd, where);
      auto x = checked_vector(b, l.size(0), where);
      banded::cholesky_solve_upper(
        l.data_ptr<double>(), l.size(0), kd, x.data_ptr<double>());
      return x;
    },
    "Solve L^T x = b with the factor of banded_cholesky_factor().",
    py::arg("factor"), py::arg("kd"), py::arg("b"));

  m.def("symmetric_band_from_pairs", &banded::symmetric_band_from_pairs,
    "Lower band storage of a symmetric matrix from (row, column, value) "
    "pairs (pairs above the diagonal are ignored). Returns (ab, kd).",
    py::arg("rows"), py::arg("cols"), py::arg("values"), py::arg("n"));
}
