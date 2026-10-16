#pragma once

#include <complex>
#include <cstdint>

namespace ttnte::linalg {

/// @brief Complex Schur decomposition A = Q T Q^* of a small dense matrix:
/// Householder reduction to Hessenberg form, then single-shift QR iteration
/// with Wilkinson shifts and deflation. Backward stable for non-normal
/// matrices (e.g. upwind transport operators with Jordan-like structure),
/// unlike an eigendecomposition.
/// @param n Matrix order.
/// @param a (n, n) row-major matrix; overwritten with the upper triangular T.
/// @param q (n, n) row-major output, the unitary Q.
/// @param max_iterations QR iterations allowed per eigenvalue.
/// @return Whether every eigenvalue converged.
bool complex_schur(int64_t n, std::complex<double>* a, std::complex<double>* q,
  int64_t max_iterations = 60);

} // namespace ttnte::linalg
