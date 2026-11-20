#pragma once

#include <cstdint>
#include <torch/extension.h>

namespace ttnte::linalg {

/// @brief Options of mgs2a().
struct MGS2Options {
  /// Cleanups continue while |q_1^T u| > eps (q_1 the first row of Q).
  double eps = 1e-10;
  /// Projections in all, at least 2: two always, then up to
  /// max_projections - 2 cleanups (Guo & Peng use "at least 2 and at most
  /// 6", i.e. m_max = 4 cleanups in their Alg. 3.2).
  int64_t max_projections = 6;
  /// x is not appended when its remainder is at most rtol ||x|| (0: only
  /// an exactly zero remainder). An addition to the algorithm, which always
  /// appends: such a remainder is rounding noise.
  double rtol = 0.0;
};

/// @brief Result of mgs2a(): x = Q^T coefficients + alpha u, u the unit
/// remainder left in x when appended.
struct MGS2Result {
  /// (r) coordinates of x on the rows of Q.
  torch::Tensor coefficients;
  /// ||x - Q^T coefficients|| (0 when x is exactly in the span).
  double alpha = 0.0;
  /// ||x|| on input.
  double norm = 0.0;
  /// Whether u is a new unit vector to append to Q. False when the
  /// remainder is zero, at most rtol ||x||, or Q already spans the space
  /// (r >= n); then the coefficients reproduce x to that accuracy.
  bool appended = false;
  /// False when the cleanups ended with |q_1^T u| > eps (an orthogonality
  /// failure; only when appended).
  bool orthogonal = true;
};

/// @brief Orthogonalize one vector against the orthonormal rows of Q by
/// classical Gram-Schmidt with reorthogonalization: two projections
/// ("twice is enough"), then adaptive cleanups while |q_1^T u| > eps, at
/// most max_projections in all (MGS2-A, Guo & Peng 2026, Alg. 3.2).
///
/// Euclidean inner product: for a weighted one, transform x (and Q) first.
/// Any dtype and device that Q and x share.
/// @param Q (r, n) basis with orthonormal rows (r may be 0).
/// @param x (n) contiguous vector; overwritten with the unit remainder u
/// when appended, otherwise with unspecified values.
/// @param options Tolerances and the projection count.
/// @throws std::runtime_error On inconsistent shapes, dtypes or devices,
/// or bad options.
MGS2Result mgs2a(
  const torch::Tensor& Q, torch::Tensor& x, const MGS2Options& options = {});

} // namespace ttnte::linalg
