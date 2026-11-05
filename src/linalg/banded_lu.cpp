#include "ttnte/linalg/banded_lu.hpp"
#include "ttnte/linalg/batched_lu.hpp"
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

int64_t lu_factor(
  double* ab, int64_t ldab, int64_t n, int64_t kl, int64_t ku, int64_t* ipiv)
{
  const int64_t kv = kl + ku;
  // A(i, j) accessor into the band storage
  auto at = [=](int64_t i, int64_t j) -> double& {
    return ab[band_offset(i, j, kl, ku, ldab)];
  };

  // The fill-in rows of the first kv columns must start at zero
  for (int64_t j = ku + 1; j < std::min(kv, n); j++) {
    for (int64_t i = kv - j; i < kl; i++) {
      ab[i + j * ldab] = 0.0;
    }
  }

  int64_t ju = 0;
  for (int64_t j = 0; j < n; j++) {
    // Zero the fill-in elements of column j + kv
    if (j + kv < n) {
      for (int64_t i = 0; i < kl; i++) {
        ab[i + (j + kv) * ldab] = 0.0;
      }
    }

    // Find the pivot within the band of column j
    const int64_t km = std::min(kl, n - 1 - j);
    int64_t p = 0;
    double pmax = std::abs(at(j, j));
    for (int64_t t = 1; t <= km; t++) {
      double v = std::abs(at(j + t, j));
      if (v > pmax) {
        pmax = v;
        p = t;
      }
    }
    ipiv[j] = j + p;
    if (at(j + p, j) == 0.0) {
      return j + 1;
    }

    // Row interchange over the columns touched by this pivot
    ju = std::max(ju, std::min(j + ku + p, n - 1));
    if (p != 0) {
      for (int64_t c = j; c <= ju; c++) {
        std::swap(at(j + p, c), at(j, c));
      }
    }

    // Compute multipliers and update the trailing band
    if (km > 0) {
      const double inv = 1.0 / at(j, j);
      for (int64_t t = 1; t <= km; t++) {
        at(j + t, j) *= inv;
      }
      for (int64_t c = j + 1; c <= ju; c++) {
        const double a = at(j, c);
        if (a != 0.0) {
          for (int64_t t = 1; t <= km; t++) {
            at(j + t, c) -= at(j + t, j) * a;
          }
        }
      }
    }
  }
  return 0;
}

void lu_solve(const double* ab, int64_t ldab, int64_t n, int64_t kl, int64_t ku,
  const int64_t* ipiv, double* b)
{
  const int64_t kv = kl + ku;
  auto at = [=](int64_t i, int64_t j) -> double {
    return ab[band_offset(i, j, kl, ku, ldab)];
  };

  // Forward substitution with L (unit lower, row interchanges applied)
  if (kl > 0) {
    for (int64_t j = 0; j < n - 1; j++) {
      const int64_t km = std::min(kl, n - 1 - j);
      const int64_t l = ipiv[j];
      if (l != j) {
        std::swap(b[l], b[j]);
      }
      const double bj = b[j];
      if (bj != 0.0) {
        for (int64_t t = 1; t <= km; t++) {
          b[j + t] -= at(j + t, j) * bj;
        }
      }
    }
  }

  // Back substitution with U (upper band of width kv)
  for (int64_t j = n - 1; j >= 0; j--) {
    b[j] /= at(j, j);
    const double bj = b[j];
    if (bj != 0.0) {
      for (int64_t i = std::max<int64_t>(0, j - kv); i < j; i++) {
        b[i] -= at(i, j) * bj;
      }
    }
  }
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
