#include "ttnte/linalg/banded_lu.hpp"
#include "ttnte/linalg/batched_lu.hpp"
#include "ttnte/linalg/lapack.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>

namespace ttnte::linalg::banded {

namespace {

void check_bands(const char* where, int64_t kl, int64_t ku)
{
  if (kl < 0 || ku < 0)
    throw utils::runtime_error(where, "kl and ku must be nonnegative");
}

} // namespace

// LAPACK (dgbtrf/dgbtrs, zgbtrf/zgbtrs): the band storage and the info
// convention are LAPACK's already; only the pivots convert (0-based int64
// here, 1-based int there).

int64_t lu_factor(
  double* ab, int64_t ldab, int64_t n, int64_t kl, int64_t ku, int64_t* ipiv)
{
  lapack::SingleThreaded single;
  const int m = static_cast<int>(n), l = static_cast<int>(kl),
            u = static_cast<int>(ku), ld = static_cast<int>(ldab);
  int* lp = lapack::pivot_buffer(n);
  int info = 0;
  dgbtrf_(&m, &m, &l, &u, ab, &ld, lp, &info);
  if (info < 0)
    throw utils::runtime_error(
      "ttnte::linalg::banded::lu_factor", "dgbtrf: bad argument");
  lapack::from_lapack(lp, n, ipiv);
  return info;
}

int64_t lu_factor(std::complex<double>* ab, int64_t ldab, int64_t n, int64_t kl,
  int64_t ku, int64_t* ipiv)
{
  lapack::SingleThreaded single;
  const int m = static_cast<int>(n), l = static_cast<int>(kl),
            u = static_cast<int>(ku), ld = static_cast<int>(ldab);
  int* lp = lapack::pivot_buffer(n);
  int info = 0;
  zgbtrf_(&m, &m, &l, &u, ab, &ld, lp, &info);
  if (info < 0)
    throw utils::runtime_error(
      "ttnte::linalg::banded::lu_factor", "zgbtrf: bad argument");
  lapack::from_lapack(lp, n, ipiv);
  return info;
}

void lu_solve(const double* ab, int64_t ldab, int64_t n, int64_t kl, int64_t ku,
  const int64_t* ipiv, double* b)
{
  const int m = static_cast<int>(n), l = static_cast<int>(kl),
            u = static_cast<int>(ku), ld = static_cast<int>(ldab), one = 1;
  int info = 0;
  dgbtrs_("N", &m, &l, &u, &one, ab, &ld, lapack::to_lapack(ipiv, n), b, &m,
    &info, 1);
}

void lu_solve(const std::complex<double>* ab, int64_t ldab, int64_t n,
  int64_t kl, int64_t ku, const int64_t* ipiv, std::complex<double>* b)
{
  const int m = static_cast<int>(n), l = static_cast<int>(kl),
            u = static_cast<int>(ku), ld = static_cast<int>(ldab), one = 1;
  int info = 0;
  zgbtrs_("N", &m, &l, &u, &one, ab, &ld, lapack::to_lapack(ipiv, n), b, &m,
    &info, 1);
}

torch::Tensor batched_lu_factor_(torch::Tensor& ab, int64_t kl, int64_t ku)
{
  check_bands("ttnte::linalg::banded::batched_lu_factor_", kl, ku);
  const int64_t ldab = ab.dim() == 3 ? ab.size(2) : 0;
  const int64_t n = ab.dim() == 3 ? ab.size(1) : 0;
  return batched_factor("ttnte::linalg::banded::batched_lu_factor_", ab,
    factor_ldab(kl, ku), [&](double* m, int64_t* piv) {
      return lu_factor(m, ldab, n, kl, ku, piv);
    });
}

void batched_lu_solve_(const torch::Tensor& ab, const torch::Tensor& ipiv,
  int64_t kl, int64_t ku, torch::Tensor& b)
{
  check_bands("ttnte::linalg::banded::batched_lu_solve_", kl, ku);
  if (ab.dim() == 3 && ab.size(2) < factor_ldab(kl, ku)) {
    throw utils::runtime_error("ttnte::linalg::banded::batched_lu_solve_",
      "ldab must be >= 2 kl + ku + 1");
  }
  const int64_t ldab = ab.dim() == 3 ? ab.size(2) : 0;
  const int64_t n = ab.dim() == 3 ? ab.size(1) : 0;
  batched_solve("ttnte::linalg::banded::batched_lu_solve_", ab, ipiv, b,
    [&](const double* m, const int64_t* piv, double* rhs) {
      lu_solve(m, ldab, n, kl, ku, piv, rhs);
    });
}

torch::Tensor dense_to_band(const torch::Tensor& a, int64_t kl, int64_t ku)
{
  auto dense = a.dim() == 2 ? a.unsqueeze(0) : a;
  dense = dense.to(torch::kFloat64).contiguous();
  const int64_t batch = dense.size(0);
  const int64_t n = dense.size(1);
  const int64_t ldab = factor_ldab(kl, ku);

  auto ab = torch::zeros({batch, n, ldab}, torch::kFloat64);
  auto acc = dense.accessor<double, 3>();
  auto bcc = ab.accessor<double, 3>();
  for (int64_t k = 0; k < batch; k++) {
    for (int64_t j = 0; j < n; j++) {
      for (int64_t i = std::max<int64_t>(0, j - ku);
        i <= std::min(n - 1, j + kl); i++) {
        bcc[k][j][band_offset(i, j, kl, ku, ldab) - j * ldab] = acc[k][i][j];
      }
    }
  }
  return ab;
}

std::pair<int64_t, int64_t> bandwidth_from_pairs(
  const torch::Tensor& rows, const torch::Tensor& cols)
{
  if (rows.numel() == 0)
    return {0, 0};
  auto diff = rows.to(torch::kInt64) - cols.to(torch::kInt64);
  return {std::max<int64_t>(0, diff.max().item<int64_t>()),
    std::max<int64_t>(0, (-diff).max().item<int64_t>())};
}

std::pair<int64_t, int64_t> bandwidth(const torch::Tensor& a)
{
  auto pattern = (a.dim() == 3 ? a.abs().sum(0) : a.abs()) != 0;
  auto nz = pattern.nonzero();
  return bandwidth_from_pairs(nz.select(1, 0), nz.select(1, 1));
}

} // namespace ttnte::linalg::banded
