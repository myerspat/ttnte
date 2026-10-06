#pragma once

#include <cstdint>

namespace ttnte::sweep {

/// @brief Local solver used for each patch's per-direction, per-group system.
enum class LocalSolverType : uint8_t {
  /// Banded LU with partial pivoting -- the natural choice for patch
  /// discontinuous (PDG) patches, whose tensor-product ordering gives a
  /// bandwidth of roughly p * N_x (2-D).
  BANDED_LU,
  /// Batched dense LU -- the natural choice for single-element (FDG) patches
  /// with (p + 1)^d unknowns.
  DENSE_LU,
};

/// @brief Configuration of the transport sweep solver.
struct SweepConfig {
  /// Local patch solver.
  LocalSolverType local_solver = LocalSolverType::BANDED_LU;
  /// Keep the full angular flux of every patch (N_Omega x G x N per patch).
  /// When false only the flux moments and the face traces needed by the
  /// sweep (interfaces and reflective faces) are stored.
  bool store_angular_flux = false;
  /// Worker threads of the task scheduler.
  int64_t num_threads = 1;
  /// Number of energy groups per sweep task (<= 0 means all groups in one
  /// task). Smaller group sets expose more concurrent tasks.
  int64_t group_set_size = 0;
  /// Relative tolerance on the scalar flux change between source
  /// iterations.
  double tol = 1e-6;
  /// Maximum number of source iterations (sweeps).
  int64_t max_iter = 1000;
  /// Scale the stopping test by (1 - rho), rho being the estimated spectral
  /// radius, so slowly converging iterations are not mistaken for converged
  /// ones (OpenSn's criterion).
  bool rho_safety = true;
  /// Faces whose upwind integral is below this fraction of the face area
  /// carry no sweep dependency for a direction.
  double face_weight_tol = 1e-12;
  /// Print iteration progress.
  bool verbose = true;
};

} // namespace ttnte::sweep
