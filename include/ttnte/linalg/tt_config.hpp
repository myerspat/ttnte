#pragma once

namespace ttnte::linalg {

/// @brief Configuration common among TT methods.
struct TTConfig {
  /// Truncation tolerance
  double eps = 1e-12;
  /// Max rank.
  int max_rank = 500;

  // =================================================================
  // Constructors
  TTConfig(double eps = 1e-12, int max_rank = 500)
    : eps(eps), max_rank(max_rank)
  {}
};

/// @brief Configuration for TT-cross.
struct CrossConfig : public TTConfig {
  /// Maximum number of sweeps.
  int nswp = 50;
  /// Rank increase each sweep.
  int kick = 4;
  /// Print progress to terminal.
  bool verbose = false;

  // =================================================================
  // Constructors
  CrossConfig(double eps = 1e-12, int max_rank = 500, int nswp = 50,
    int kick = 4, bool verbose = false)
    : TTConfig(eps, max_rank), nswp(nswp), kick(kick), verbose(verbose)
  {}
};

} // namespace ttnte::linalg
