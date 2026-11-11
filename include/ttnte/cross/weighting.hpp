#pragma once

namespace ttnte::cross {

/// @brief Which factors of the L2 weighting a sampler applies to the fibers
/// it stores and measures. A fiber of direction d and group g is stored as
///   a_d e_g R_P psi_{d,g}
/// on each patch P, with a_d = sqrt(w_d), e_g = 1 / sqrt(du_g) and R_P the
/// upper Cholesky factor of the patch mass matrix (M_P = R_P^T R_P); a
/// disabled factor is the identity. With all three the Euclidean norm of the
/// stored fibers is the L2 norm of the angular flux in angle, energy (per
/// unit lethargy) and space.
struct Weighting {
  /// sqrt(w_d) per direction.
  bool angle = true;
  /// 1 / sqrt(du_g) per group. Needs the problem's lethargy widths; without
  /// them energy stays unweighted.
  bool energy = true;
  /// R_P per patch. Needs the problem's mass bands; without them space stays
  /// unweighted.
  bool space = true;

  /// @return Whether any factor is enabled.
  bool any() const noexcept { return angle || energy || space; }
};

} // namespace ttnte::cross
