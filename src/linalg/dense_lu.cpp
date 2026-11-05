#include "ttnte/linalg/dense_lu.hpp"
#include "ttnte/linalg/batched_lu.hpp"
#include "ttnte/utils/exception.hpp"
#include <cmath>
#include <utility>

namespace ttnte::linalg::dense {

int64_t lu_factor(double* a, int64_t n, int64_t* ipiv)
{
  for (int64_t j = 0; j < n; j++) {
    // Pivot: largest magnitude in column j at or below the diagonal
    int64_t p = j;
    double pmax = std::abs(a[j * n + j]);
    for (int64_t i = j + 1; i < n; i++) {
      double v = std::abs(a[i * n + j]);
      if (v > pmax) {
        pmax = v;
        p = i;
      }
    }
    ipiv[j] = p;
    if (a[p * n + j] == 0.0) {
      return j + 1;
    }
    if (p != j) {
      for (int64_t c = 0; c < n; c++) {
        std::swap(a[p * n + c], a[j * n + c]);
      }
    }

    // Multipliers and trailing update
    const double inv = 1.0 / a[j * n + j];
    for (int64_t i = j + 1; i < n; i++) {
      double& l = a[i * n + j];
      l *= inv;
      if (l != 0.0) {
        for (int64_t c = j + 1; c < n; c++) {
          a[i * n + c] -= l * a[j * n + c];
        }
      }
    }
  }
  return 0;
}

void lu_solve(const double* a, int64_t n, const int64_t* ipiv, double* b)
{
  // Row interchanges, then L (unit lower) and U substitutions
  for (int64_t j = 0; j < n; j++) {
    if (ipiv[j] != j) {
      std::swap(b[j], b[ipiv[j]]);
    }
  }
  for (int64_t i = 1; i < n; i++) {
    double s = b[i];
    for (int64_t c = 0; c < i; c++) {
      s -= a[i * n + c] * b[c];
    }
    b[i] = s;
  }
  for (int64_t i = n - 1; i >= 0; i--) {
    double s = b[i];
    for (int64_t c = i + 1; c < n; c++) {
      s -= a[i * n + c] * b[c];
    }
    b[i] = s / a[i * n + i];
  }
}

torch::Tensor batched_lu_factor_(torch::Tensor& a)
{
  if (a.dim() == 3 && a.size(1) != a.size(2)) {
    throw utils::runtime_error(
      "ttnte::linalg::dense::batched_lu_factor_", "Matrices must be square");
  }
  const int64_t n = a.dim() == 3 ? a.size(1) : 0;
  return batched_factor("ttnte::linalg::dense::batched_lu_factor_", a, n,
    [n](double* m, int64_t* piv) { return lu_factor(m, n, piv); });
}

void batched_lu_solve_(
  const torch::Tensor& a, const torch::Tensor& ipiv, torch::Tensor& b)
{
  if (a.dim() == 3 && a.size(1) != a.size(2)) {
    throw utils::runtime_error(
      "ttnte::linalg::dense::batched_lu_solve_", "Matrices must be square");
  }
  const int64_t n = a.dim() == 3 ? a.size(1) : 0;
  batched_solve("ttnte::linalg::dense::batched_lu_solve_", a, ipiv, b,
    [n](const double* m, const int64_t* piv, double* rhs) {
      lu_solve(m, n, piv, rhs);
    });
}

} // namespace ttnte::linalg::dense
