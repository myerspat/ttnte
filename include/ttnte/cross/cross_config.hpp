#pragma once

#include "ttnte/tensor/layout.hpp"
#include <cstdint>
#include <vector>

namespace ttnte::cross {

/// @brief Options of the sweep-sampled TT-cross of one transport solve.
struct CrossConfig {
  /// Order and grouping of (polar, azimuthal, space, energy) into TT cores.
  /// Splitting angle or energy into cores changes which fibers are swept;
  /// space is always sampled whole, so splitting x/y/z only shapes the
  /// per-patch output TTs. Empty: polar|azimuthal|space|energy (polar|space|
  /// energy in 1-D).
  tensor::Layout layout;
  /// Relative accuracy of the approximation (L2-weighted when `weighted`).
  double tol = 1e-4;
  /// Approximate the L2-weighted angular flux sqrt(w_d) (R psi)/sqrt(du_g),
  /// M = R^T R the patch mass matrices and du the lethargy widths, so that
  /// pivots, truncation and the hold-out error are L2 quantities.
  bool weighted = true;
  /// (G) lethargy widths for the energy weighting; empty means uniform.
  std::vector<double> lethargy_widths;
  /// Keep the TT of the angular flux after each transport solve.
  bool keep_tt = false;
  /// Hold-out fibers swept per check.
  int64_t holdout = 16;
  /// Acceptance of the hold-out error: holdout_factor * tol (fiber errors
  /// overstate moment errors).
  double holdout_factor = 10.0;
  /// Maximum forward/backward sweeps of the cross per transport solve.
  int64_t max_sweeps = 8;
  /// Cap on every TT rank (<= 0: none).
  int64_t max_rank = 0;
  /// Seed of the hold-out sampling.
  uint64_t seed = 0;
};

} // namespace ttnte::cross
