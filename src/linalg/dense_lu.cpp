#include "ttnte/linalg/dense_lu.hpp"
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
  TORCH_CHECK(a.dim() == 3 && a.size(1) == a.size(2) &&
                a.scalar_type() == torch::kFloat64 && a.is_contiguous() &&
                a.device().is_cpu(),
    "ttnte::linalg::dense::batched_lu_factor_: expected a contiguous CPU "
    "float64 tensor of shape (batch, n, n)");
  const int64_t batch = a.size(0);
  const int64_t n = a.size(1);
  auto ipiv = torch::empty({batch, n}, torch::kInt64);
  double* ap = a.data_ptr<double>();
  int64_t* pp = ipiv.data_ptr<int64_t>();
  for (int64_t k = 0; k < batch; k++) {
    int64_t info = lu_factor(ap + k * n * n, n, pp + k * n);
    if (info != 0) {
      throw utils::runtime_error("ttnte::linalg::dense::batched_lu_factor_",
        "Matrix " + std::to_string(k) +
          " of the batch is singular (zero pivot at column " +
          std::to_string(info - 1) + ")");
    }
  }
  return ipiv;
}

void batched_lu_solve_(
  const torch::Tensor& a, const torch::Tensor& ipiv, torch::Tensor& b)
{
  TORCH_CHECK(b.dim() == 2 && b.scalar_type() == torch::kFloat64 &&
                b.is_contiguous() && b.size(0) == a.size(0) &&
                b.size(1) == a.size(1),
    "ttnte::linalg::dense::batched_lu_solve_: expected contiguous float64 "
    "right-hand sides of shape (batch, n)");
  const int64_t batch = a.size(0);
  const int64_t n = a.size(1);
  const double* ap = a.data_ptr<double>();
  const int64_t* pp = ipiv.data_ptr<int64_t>();
  double* bp = b.data_ptr<double>();
  for (int64_t k = 0; k < batch; k++) {
    lu_solve(ap + k * n * n, n, pp + k * n, bp + k * n);
  }
}

} // namespace ttnte::linalg::dense
