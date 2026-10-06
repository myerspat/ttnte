#include "ttnte/linalg/banded_lu.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>

namespace ttnte::linalg::banded {

int64_t lu_factor(
  double* ab, int64_t ldab, int64_t n, int64_t kl, int64_t ku, int64_t* ipiv)
{
  const int64_t kv = kl + ku;
  // A(i, j) accessor into the band storage
  auto at = [ab, ldab, kv](int64_t i, int64_t j) -> double& {
    return ab[(kv + i - j) + j * ldab];
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
  auto at = [ab, ldab, kv](int64_t i, int64_t j) -> double {
    return ab[(kv + i - j) + j * ldab];
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
  TORCH_CHECK(ab.dim() == 3 && ab.scalar_type() == torch::kFloat64 &&
                ab.is_contiguous() && ab.device().is_cpu(),
    "ttnte::linalg::banded::batched_lu_factor_: expected a contiguous CPU "
    "float64 tensor of shape (batch, n, ldab)");
  const int64_t batch = ab.size(0);
  const int64_t n = ab.size(1);
  const int64_t ldab = ab.size(2);
  TORCH_CHECK(ldab >= factor_ldab(kl, ku),
    "ttnte::linalg::banded::batched_lu_factor_: ldab must be >= 2kl+ku+1");

  auto ipiv = torch::empty({batch, n}, torch::kInt64);
  double* abp = ab.data_ptr<double>();
  int64_t* pp = ipiv.data_ptr<int64_t>();
  for (int64_t k = 0; k < batch; k++) {
    int64_t info = lu_factor(abp + k * n * ldab, ldab, n, kl, ku, pp + k * n);
    if (info != 0) {
      throw utils::runtime_error("ttnte::linalg::banded::batched_lu_factor_",
        "Matrix " + std::to_string(k) +
          " of the batch is singular (zero "
          "pivot at column " +
          std::to_string(info - 1) + ")");
    }
  }
  return ipiv;
}

void batched_lu_solve_(const torch::Tensor& ab, const torch::Tensor& ipiv,
  int64_t kl, int64_t ku, torch::Tensor& b)
{
  TORCH_CHECK(b.dim() == 2 && b.scalar_type() == torch::kFloat64 &&
                b.is_contiguous() && b.size(0) == ab.size(0) &&
                b.size(1) == ab.size(1),
    "ttnte::linalg::banded::batched_lu_solve_: expected contiguous float64 "
    "right-hand sides of shape (batch, n)");
  const int64_t batch = ab.size(0);
  const int64_t n = ab.size(1);
  const int64_t ldab = ab.size(2);
  const double* abp = ab.data_ptr<double>();
  const int64_t* pp = ipiv.data_ptr<int64_t>();
  double* bp = b.data_ptr<double>();
  for (int64_t k = 0; k < batch; k++) {
    lu_solve(abp + k * n * ldab, ldab, n, kl, ku, pp + k * n, bp + k * n);
  }
}

torch::Tensor dense_to_band(const torch::Tensor& a, int64_t kl, int64_t ku)
{
  auto dense = a.dim() == 2 ? a.unsqueeze(0) : a;
  dense = dense.to(torch::kFloat64).contiguous();
  const int64_t batch = dense.size(0);
  const int64_t n = dense.size(1);
  const int64_t ldab = factor_ldab(kl, ku);
  const int64_t kv = kl + ku;

  auto ab = torch::zeros({batch, n, ldab}, torch::kFloat64);
  auto acc = dense.accessor<double, 3>();
  auto bcc = ab.accessor<double, 3>();
  for (int64_t k = 0; k < batch; k++) {
    for (int64_t j = 0; j < n; j++) {
      for (int64_t i = std::max<int64_t>(0, j - ku);
        i <= std::min(n - 1, j + kl); i++) {
        bcc[k][j][kv + i - j] = acc[k][i][j];
      }
    }
  }
  return ab;
}

std::pair<int64_t, int64_t> bandwidth(const torch::Tensor& a)
{
  auto pattern = (a.dim() == 3 ? a.abs().sum(0) : a.abs()) != 0;
  auto nz = pattern.nonzero();
  if (nz.size(0) == 0) {
    return {0, 0};
  }
  auto diff = nz.select(1, 0) - nz.select(1, 1);
  int64_t kl = std::max<int64_t>(0, diff.max().item<int64_t>());
  int64_t ku = std::max<int64_t>(0, (-diff).max().item<int64_t>());
  return {kl, ku};
}

} // namespace ttnte::linalg::banded
