#pragma once

#include <cstdint>
#include <torch/extension.h>

namespace ttnte::linalg::dense {

// =================================================================
// Small dense LU with partial pivoting
//
// Raw-array kernels for many small systems (e.g. one per element and
// direction in a transport sweep), where a framework call per system would
// cost far more than the arithmetic. Matrices are row-major n x n arrays,
// A(i, j) = a[i * n + j]; the algorithm is LAPACK's unblocked dgetf2/dgetrs.

/// @brief Factor one matrix in place, P A = L U.
/// @param a Row-major n x n matrix, overwritten with L (unit, below the
/// diagonal) and U.
/// @param n Matrix order.
/// @param ipiv Output pivot rows (0-based), length n.
/// @return 0 on success, or j + 1 if U(j, j) is exactly zero.
int64_t lu_factor(double* a, int64_t n, int64_t* ipiv);

/// @brief Solve A x = b with a matrix factored by lu_factor(). `b` is
/// overwritten with x.
void lu_solve(const double* a, int64_t n, const int64_t* ipiv, double* b);

/// @brief Factor a batch of matrices in place.
/// @param a (batch, n, n) contiguous CPU float64 matrices.
/// @return The (batch, n) int64 pivots.
/// @throws std::runtime_error If any matrix is singular.
torch::Tensor batched_lu_factor_(torch::Tensor& a);

/// @brief Solve a batch of factored systems in place.
/// @param a (batch, n, n) factored matrices.
/// @param ipiv (batch, n) pivots from batched_lu_factor_().
/// @param b (batch, n) right-hand sides, overwritten with the solutions.
void batched_lu_solve_(
  const torch::Tensor& a, const torch::Tensor& ipiv, torch::Tensor& b);

} // namespace ttnte::linalg::dense
