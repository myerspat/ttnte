#include "ttnte/linalg/banded_cholesky.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>

namespace ttnte::linalg::banded {

int64_t cholesky_factor(double* ab, int64_t n, int64_t kd)
{
  const int64_t ld = kd + 1;
  for (int64_t j = 0; j < n; j++) {
    double* col = ab + j * ld;
    const double diag = col[0];
    if (!(diag > 0.0))
      return j + 1;
    const double ljj = std::sqrt(diag);
    col[0] = ljj;
    const int64_t m = std::min(kd, n - 1 - j);
    for (int64_t i = 1; i <= m; i++)
      col[i] /= ljj;
    // Rank-one update of the trailing band: A(j+a, j+b) -= L(j+a, j) L(j+b, j)
    for (int64_t b = 1; b <= m; b++) {
      double* target = ab + (j + b) * ld;
      const double lb = col[b];
      for (int64_t a = b; a <= m; a++)
        target[a - b] -= col[a] * lb;
    }
  }
  return 0;
}

void cholesky_apply_upper(const double* ab, int64_t n, int64_t kd, double* x)
{
  // (L^T x)_j = sum_{i >= j} L(i, j) x_i, top to bottom keeps x_i (i > j)
  const int64_t ld = kd + 1;
  for (int64_t j = 0; j < n; j++) {
    const double* col = ab + j * ld;
    const int64_t m = std::min(kd, n - 1 - j);
    double sum = col[0] * x[j];
    for (int64_t i = 1; i <= m; i++)
      sum += col[i] * x[j + i];
    x[j] = sum;
  }
}

void cholesky_solve_upper(const double* ab, int64_t n, int64_t kd, double* b)
{
  const int64_t ld = kd + 1;
  for (int64_t j = n - 1; j >= 0; j--) {
    const double* col = ab + j * ld;
    const int64_t m = std::min(kd, n - 1 - j);
    double sum = b[j];
    for (int64_t i = 1; i <= m; i++)
      sum -= col[i] * b[j + i];
    b[j] = sum / col[0];
  }
}

std::pair<torch::Tensor, int64_t> symmetric_band_from_pairs(
  const torch::Tensor& rows, const torch::Tensor& cols,
  const torch::Tensor& values, int64_t n)
{
  auto r = rows.to(torch::kInt64).contiguous();
  auto c = cols.to(torch::kInt64).contiguous();
  auto v = values.to(torch::kFloat64).contiguous();
  const int64_t P = r.size(0);
  const int64_t* rp = r.data_ptr<int64_t>();
  const int64_t* cp = c.data_ptr<int64_t>();
  const double* vp = v.data_ptr<double>();
  int64_t kd = 0;
  for (int64_t p = 0; p < P; p++)
    kd = std::max(kd, rp[p] - cp[p]);
  auto ab = torch::zeros({n, kd + 1}, torch::kFloat64);
  double* a = ab.data_ptr<double>();
  for (int64_t p = 0; p < P; p++) {
    if (rp[p] < cp[p])
      continue;
    a[cp[p] * (kd + 1) + (rp[p] - cp[p])] += vp[p];
  }
  return {ab, kd};
}

} // namespace ttnte::linalg::banded
