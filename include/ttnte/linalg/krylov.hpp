#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace ttnte::linalg::krylov {

/// @brief Matrix-free operator y = A x on raw vectors of a fixed length.
using LinearOperator = std::function<void(const double* x, double* y)>;

/// @brief Outcome of a Krylov solve.
struct KrylovResult {
  /// Operator applications performed.
  int64_t num_applies = 0;
  /// Final relative residual ||b - A x|| / ||b||: the true residual at a
  /// restart, the Arnoldi estimate when converging within a cycle.
  double relative_residual = 0.0;
  bool converged = false;
};

/// @brief Restarted GMRES(m) with modified Gram-Schmidt and Givens
/// rotations, on plain double vectors.
///
/// Matrix-free and framework-free, so any operator whose application is
/// expensive (e.g. a transport sweep per product) can be solved without
/// forming it. The true residual is recomputed at every restart; convergence
/// within a cycle is accepted on the Arnoldi residual estimate.
/// @param apply y = A x.
/// @param b Right-hand side (length n).
/// @param x Initial guess on entry, solution on exit (length n).
/// @param restart Krylov subspace size m before restarting (>= 1).
/// @param tol Relative residual tolerance.
/// @param max_applies Maximum number of operator applications.
KrylovResult gmres(const LinearOperator& apply, const std::vector<double>& b,
  std::vector<double>& x, int64_t restart, double tol, int64_t max_applies);

} // namespace ttnte::linalg::krylov
