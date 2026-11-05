#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace ttnte::linalg::krylov {

/// @brief Matrix-free operator y = A x on raw vectors of a fixed length.
using LinearOperator = std::function<void(const double* x, double* y)>;

/// @brief Global sum of a rank-local partial reduction (e.g. an MPI
/// allreduce), applied to every inner product of a distributed vector.
using Reducer = std::function<double(double)>;

/// @brief Called before every operator application with the current relative
/// residual: the Arnoldi estimate, or 1 before a true-residual product
/// (`true_residual`, at the start of every cycle), which should be applied
/// accurately. Lets an inexact operator relax its accuracy as the residual
/// drops (Simoncini and Szyld, SISC 25, 2003).
using Progress =
  std::function<void(double relative_residual, bool true_residual)>;

/// @brief Scratch of gmres(): the Krylov basis, Hessenberg matrix and
/// rotations, kept between calls so a solver called many times (e.g. one
/// local solve per direction and group) does not reallocate them.
struct Workspace {
  std::vector<double> V, H, cs, sn, g, w, r, y;
};

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
/// forming it. The true residual is recomputed at every restart (without an
/// application when the initial guess is zero); convergence within a cycle
/// is accepted on the Arnoldi residual estimate.
/// @param apply y = A x.
/// @param b Right-hand side (length n).
/// @param x Initial guess on entry, solution on exit (length n).
/// @param restart Krylov subspace size m before restarting (>= 1).
/// @param tol Relative residual tolerance.
/// @param max_applies Maximum number of operator applications (at least one
/// is made unless the initial guess is zero).
/// @param reduce Optional global sum of inner products for vectors
/// distributed across ranks (each rank passes its local slice); empty for
/// a serial solve.
/// @param progress Optional callback before every application.
KrylovResult gmres(const LinearOperator& apply, const std::vector<double>& b,
  std::vector<double>& x, int64_t restart, double tol, int64_t max_applies,
  const Reducer& reduce = {}, const Progress& progress = {});

/// @brief gmres() with caller-owned scratch (see Workspace).
KrylovResult gmres(const LinearOperator& apply, const std::vector<double>& b,
  std::vector<double>& x, int64_t restart, double tol, int64_t max_applies,
  Workspace& work, const Reducer& reduce = {}, const Progress& progress = {});

} // namespace ttnte::linalg::krylov
