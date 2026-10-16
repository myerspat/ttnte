#pragma once

#include <cstdint>
#include <torch/extension.h>

namespace ttnte::linalg::banded {

// =================================================================
// Banded LU with partial pivoting
//
// Storage follows LAPACK's general band format (dgbtrf/dgbtrs), in
// column-major order: a band matrix A with `kl` sub- and `ku`
// super-diagonals is held in an array `ab` with leading dimension
// `ldab >= 2 * kl + ku + 1`, where A(i, j) lives at
//   ab[(kl + ku + i - j) + j * ldab]
// for max(0, j - ku) <= i <= min(n - 1, j + kl). The first `kl` rows hold
// the fill-in created by row interchanges. As a torch tensor, one matrix is
// a contiguous (n, ldab) array whose row j is column j of the band.
//
// Self-contained (no LAPACK link): torch's bundled LAPACK does not export
// the banded routines, and the unblocked algorithm (dgbtf2) is adequate for
// the bandwidths of IGA patch matrices.

/// @return The leading dimension needed to factor a band with `kl`/`ku`
/// sub-/super-diagonals.
inline int64_t factor_ldab(int64_t kl, int64_t ku) noexcept
{
  return 2 * kl + ku + 1;
}

/// @brief Factor one band matrix in place (dgbtf2 semantics).
/// @param ab Band storage, column-major with leading dimension `ldab`.
/// @param ldab Leading dimension (>= 2 * kl + ku + 1).
/// @param n Matrix order.
/// @param kl Number of sub-diagonals.
/// @param ku Number of super-diagonals.
/// @param ipiv Output pivot indices (0-based), length n.
/// @return 0 on success, or j + 1 if U(j, j) is exactly zero.
int64_t lu_factor(
  double* ab, int64_t ldab, int64_t n, int64_t kl, int64_t ku, int64_t* ipiv);

/// @brief Solve A x = b for one right-hand side with a factored band matrix
/// (dgbtrs, no transpose). `b` is overwritten with x.
void lu_solve(const double* ab, int64_t ldab, int64_t n, int64_t kl, int64_t ku,
  const int64_t* ipiv, double* b);

/// @brief Factor a batch of band matrices in place.
/// @param ab (batch, n, ldab) contiguous float64 band storage.
/// @param kl Number of sub-diagonals.
/// @param ku Number of super-diagonals.
/// @return The (batch, n) int64 pivots.
/// @throws std::runtime_error If any matrix is singular.
torch::Tensor batched_lu_factor_(torch::Tensor& ab, int64_t kl, int64_t ku);

/// @brief Solve a batch of factored band systems in place.
/// @param ab (batch, n, ldab) factored band storage.
/// @param ipiv (batch, n) pivots from batched_lu_factor_().
/// @param b (batch, n) right-hand sides, overwritten with the solutions.
void batched_lu_solve_(const torch::Tensor& ab, const torch::Tensor& ipiv,
  int64_t kl, int64_t ku, torch::Tensor& b);

/// @brief Pack dense matrices into band storage with fill-in rows.
/// @param a (batch, n, n) dense matrices.
/// @return (batch, n, 2 * kl + ku + 1) band storage.
torch::Tensor dense_to_band(const torch::Tensor& a, int64_t kl, int64_t ku);

/// @brief Measure the bandwidth of the nonzero pattern of a matrix.
/// @param a (n, n) or (batch, n, n) dense matrix.
/// @return (kl, ku).
std::pair<int64_t, int64_t> bandwidth(const torch::Tensor& a);

} // namespace ttnte::linalg::banded
