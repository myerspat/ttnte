#include "ttnte/linalg/banded_cholesky.hpp"
#include "ttnte/linalg/banded_lu.hpp"
#include "ttnte/linalg/lapack.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>

namespace ttnte::linalg::banded {

// LAPACK/BLAS: the (n, kd + 1) storage, column j holding A(j + i, j) at i,
// is LAPACK's lower band storage with ldab = kd + 1.

int64_t cholesky_factor(double* ab, int64_t n, int64_t kd)
{
  lapack::SingleThreaded single;
  const int m = static_cast<int>(n), k = static_cast<int>(kd), ld = k + 1;
  int info = 0;
  dpbtrf_("L", &m, &k, ab, &ld, &info, 1);
  if (info < 0)
    throw utils::runtime_error(
      "ttnte::linalg::banded::cholesky_factor", "dpbtrf: bad argument");
  return info;
}

void cholesky_apply_upper(const double* ab, int64_t n, int64_t kd, double* x)
{
  const int m = static_cast<int>(n), k = static_cast<int>(kd), ld = k + 1,
            one = 1;
  dtbmv_("L", "T", "N", &m, &k, ab, &ld, x, &one, 1, 1, 1);
}

void cholesky_solve_upper(const double* ab, int64_t n, int64_t kd, double* b)
{
  const int m = static_cast<int>(n), k = static_cast<int>(kd), ld = k + 1,
            one = 1;
  dtbsv_("L", "T", "N", &m, &k, ab, &ld, b, &one, 1, 1, 1);
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
  for (int64_t p = 0; p < P; p++) {
    if (rp[p] < 0 || rp[p] >= n || cp[p] < 0 || cp[p] >= n)
      throw utils::runtime_error(
        "ttnte::linalg::banded::symmetric_band_from_pairs",
        "A pair index is out of range");
  }
  const int64_t kd = bandwidth_from_pairs(r, c).first;
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
