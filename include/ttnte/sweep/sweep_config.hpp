#pragma once

#include "ttnte/cross/cross_config.hpp"

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
  /// Kronecker-sum solver (see KroneckerSolver) on axis-aligned affine
  /// patches, sharing one Schur factorization per patch among all directions
  /// and groups; other patches fall back to BANDED_LU.
  SYLVESTER,
  /// SYLVESTER on affine patches; on curved ones GMRES on the matrix-free
  /// sum-factorized operator, preconditioned by the patch's Kronecker-sum
  /// parametric twin (see IterativeSolver). No factorization is stored.
  /// Experimental: the averaged-metric preconditioner weakens as curved
  /// patches grow, and unconverged solves fall back to LU.
  ITERATIVE,
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

/// @brief How each transport solve (one application of the swept operator)
/// is carried out.
enum class TransportSolve : uint8_t {
  /// Sweep every (direction, group) pair.
  EXACT,
  /// Sweep a few pairs chosen by a TT-cross and contract the moments and
  /// lagged traces from the resulting low-rank angular flux.
  CROSS,
};

/// @brief Configuration of the transport sweep solver.
struct SweepConfig {
  /// Local patch solver.
  LocalSolverType local_solver = LocalSolverType::BANDED_LU;
  /// Keep the full angular flux of every patch (N_Omega x G x N per patch).
  /// When false only the flux moments and the face traces needed by the
  /// sweep (interfaces and reflective faces) are stored.
  bool store_angular_flux = false;
  /// ITERATIVE: relative tolerance, restart length and maximum operator
  /// applications of the local GMRES (a solve that does not converge falls
  /// back to LU).
  double local_tol = 1e-11;
  int64_t local_restart = 20;
  int64_t local_max_iter = 100;
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
  /// Solve each within-group-set GMRES problem for the correction e to the
  /// current iterate x0: one transport solve of the full source gives
  /// y0 = T(q + S x0) and the residual r0 = y0 - x0, GMRES solves
  /// (I - T S) e = r0 to ||r|| <= tol ||y0||, and x0 + e is accepted without
  /// a final sweep (y0 itself when r0 already meets the tolerance). The
  /// first solve of each group set uses the standard form to measure
  /// ||b|| / ||x||, which keeps the tolerances relative to ||b||. One
  /// transport solve per group set per outer instead of three, and with CROSS
  /// the Krylov products only need accuracy relative to ||y0|| / ||r0||, which
  /// grows as the outer iteration converges. Storing the angular flux or
  /// keeping the cross's TT costs one more transport solve of the accepted
  /// iterate per group set.
  bool correction_form = false;
  /// Correction form only: transport just the change of the input (sources
  /// and lagged traces) since the previous outer iteration, y0 = y_prev +
  /// T(input - input_prev). The change is solved to the same absolute
  /// accuracy as y0, which relative to its (small) size is crude: a cross
  /// keeps low ranks as the outer iteration converges. A full transport
  /// solve is done when the change exceeds incremental_switch times the
  /// input.
  bool incremental_transport = false;
  double incremental_switch = 0.3;
  /// Return a Solution (each patch's flux compressed into a TT at 1e-14)
  /// from solve_fixed_source() and solve_eigenvalue(); false returns none
  /// and skips the compression.
  bool build_solution = true;
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
  /// With verbose, also print the within-group-set iterations (residual,
  /// group set, elapsed time) once at least this many seconds passed since
  /// the last progress line (0: every iteration, < 0: never). Also prints
  /// the memory held by the solver after setup.
  double progress_seconds = 30.0;
  /// Transport solve of every sweep.
  TransportSolve transport = TransportSolve::EXACT;
  /// Options of the TT-cross (TransportSolve::CROSS).
  cross::CrossConfig cross;
};

} // namespace ttnte::sweep
