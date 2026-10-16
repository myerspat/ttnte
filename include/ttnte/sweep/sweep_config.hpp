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

/// @brief How sweep units are grouped into scheduler tasks.
enum class TaskGranularity : uint8_t {
  /// One task per unit, angle set and group set: the most concurrency, for
  /// a few large units (PDG patches).
  UNIT,
  /// One task per angle set and group set sweeping every local unit in
  /// upwind order (the AAH-style chunk): the least scheduling overhead, for
  /// many small units (FDG elements).
  SUBDOMAIN,
};

/// @brief Coupling of group sets in the across-group-set iteration.
enum class GroupIteration : uint8_t {
  /// Every group set sees the previous pass's flux of the other sets
  /// (concurrent-friendly, slow with strong group coupling).
  JACOBI,
  /// Group sets are solved in energy order, each seeing the latest flux of
  /// the sets already solved: downscatter is resolved in one pass.
  GAUSS_SEIDEL,
};

/// @brief Solver of the within-group-set problem
/// (I - D L^{-1} M S_ss) phi_s = D L^{-1} q_s, one sweep per product.
enum class InnerSolver : uint8_t {
  /// Fixed-point (source) iteration.
  RICHARDSON,
  /// Restarted GMRES on the moments and the lagged face traces.
  GMRES,
};

/// @brief Configuration of the transport sweep solver.
struct SweepConfig {
  /// Local patch solver.
  LocalSolverType local_solver = LocalSolverType::BANDED_LU;
  /// Keep the full angular flux of every patch (N_Omega x G x N per patch).
  /// When false only the flux moments and the face traces needed by the
  /// sweep (interfaces and reflective faces) are stored.
  bool store_angular_flux = false;
  /// Factor every (direction, group) local system once and reuse the
  /// factors in later sweeps (the operators do not change between source
  /// iterations). Trades memory -- N_Omega * G * (N * 2kl+ku+1 or N^2)
  /// doubles per unit, see SweepDriver::factorization_bytes() -- for
  /// skipping assembly and factorization after the first sweep.
  bool cache_factorizations = false;
  /// Grouping of units into tasks (overridden by an explicit chunk map
  /// passed to SweepDriver::setup()).
  TaskGranularity task_granularity = TaskGranularity::UNIT;
  /// Worker threads of the task scheduler.
  int64_t num_threads = 1;
  /// Directions swept by one task (<= 0: a whole angle set). Smaller blocks
  /// give independent copies of the sweep DAG and so more concurrent tasks,
  /// at the cost of more, shorter tasks (16-32 was best on C5G7 with 64-96
  /// threads).
  int64_t directions_per_task = 32;
  /// Number of energy groups per sweep task (<= 0 means all groups in one
  /// task). Smaller group sets expose more concurrent tasks.
  int64_t group_set_size = 0;
  /// Coupling of group sets.
  GroupIteration group_iteration = GroupIteration::GAUSS_SEIDEL;
  /// Within-group-set solver.
  InnerSolver inner_solver = InnerSolver::GMRES;
  /// Within-group-set tolerance: GMRES relative residual or Richardson
  /// relative scalar flux change (<= 0 uses 0.1 * tol).
  double inner_tol = 0.0;
  /// Inexact-Newton style forcing of the within-group-set tolerance (<= 0
  /// disables it): outer iteration k solves to max(floor, inner_forcing *
  /// min(e_{k-1}, 1)), e_{k-1} being the previous outer scalar flux change
  /// (taken as 1 on the first outer) and floor the tolerance above.
  double inner_forcing = 0.0;
  /// Maximum sweeps of one within-group-set solve (1 with RICHARDSON and
  /// JACOBI is classic source iteration).
  int64_t inner_max_iter = 1000;
  /// GMRES restart length.
  int64_t gmres_restart = 30;
  /// Relative tolerance on the scalar flux change between outer iterations
  /// (across-group-set passes for fixed source, power iterations for
  /// k-eigenvalue).
  double tol = 1e-6;
  /// Maximum number of outer iterations.
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
