#pragma once

#include <cstdint>
#include <torch/extension.h>

namespace ttnte::linalg::banded {

// Symmetric positive definite band matrices with `kd` sub-diagonals are
// held in lower band storage: A(i, j), j <= i <= j + kd, lives at
// ab[j * (kd + 1) + (i - j)]. As a torch tensor, one matrix is a contiguous
// (n, kd + 1) array whose row j is column j of the lower band.

/// @brief Cholesky factorization A = L L^T in place (L overwrites the lower
/// band).
/// @return 0 on success, or j + 1 if the leading minor of order j + 1 is
/// not positive definite.
int64_t cholesky_factor(double* ab, int64_t n, int64_t kd);

/// @brief x <- L^T x with the factor of cholesky_factor().
void cholesky_apply_upper(const double* ab, int64_t n, int64_t kd, double* x);

/// @brief Solve L^T x = b in place (b is overwritten with x).
void cholesky_solve_upper(const double* ab, int64_t n, int64_t kd, double* b);

/// @brief Lower band storage of a symmetric matrix given by a pair list.
/// @param rows (P) row indices.
/// @param cols (P) column indices.
/// @param values (P) values (pairs above the diagonal are ignored).
/// @param n Matrix order.
/// @return (n, kd + 1) band storage and kd.
std::pair<torch::Tensor, int64_t> symmetric_band_from_pairs(
  const torch::Tensor& rows, const torch::Tensor& cols,
  const torch::Tensor& values, int64_t n);

} // namespace ttnte::linalg::banded
