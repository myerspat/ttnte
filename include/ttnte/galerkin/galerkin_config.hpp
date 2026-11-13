#pragma once

#include "ttnte/sampling/weighting.hpp"
#include <cstdint>

namespace ttnte::galerkin {

/// @brief The first snapshots of a greedy Galerkin run.
enum class GalerkinStart : uint8_t {
  /// per_iter random directions (Guo & Peng: "an initial set of p angular
  /// directions is selected at random").
  RANDOM,
  /// The cross's initial directions (SamplingProblem::initial_directions), so
  /// both samplers start from the same fibers.
  INITIAL_DIRECTIONS,
};

/// @brief What a run's first full solve compares its scalar flux with in
/// the change test (3.9).
enum class GalerkinPhiOld : uint8_t {
  /// PREVIOUS_RUN under the RICHARDSON inner solver, WITHIN_RUN under GMRES
  /// (resolved by the sweep solver at setup).
  AUTO,
  /// The previous run's scalar flux, carried across runs: Guo & Peng's
  /// phi_old := phi^(n-1) under source iteration, where the previous run's
  /// flux is this run's input (no DSA). The first full solve may stop.
  PREVIOUS_RUN,
  /// Reset at each run: (3.9) compares successive full solves of the same
  /// run, so a run does at least two. Han et al. (2026, Sec. 3.1, 5.1)
  /// under GMRES, where each run's input is a Krylov vector and the
  /// previous run's flux is unrelated.
  WITHIN_RUN,
};

/// @brief Options of the residual-greedy Galerkin transport solve (Guo &
/// Peng, 2026, Alg. 3.1-3.2, multigroup by angular augmentation over a group
/// set's groups).
struct GalerkinConfig {
  /// Directions swept per greedy iteration (p).
  int64_t per_iter = 1;
  /// Random unswept candidates scored per iteration (q > p).
  int64_t candidates = 8;
  /// First snapshots.
  GalerkinStart start = GalerkinStart::RANDOM;
  /// Relative accuracy of a run: the residual test (3.7) is
  ///   max_j ||r_j||* <= tol * RMS_j ||b_j||*
  /// over the q candidates (b_j their right-hand sides), and the change test
  /// (3.9) ||phi - phi_old||_w <= tol ||phi||_w. Inside the sweep solver it
  /// is set per transport solve instead (SweepConfig::cross_inner_factor and
  /// cross_relax, as for the cross).
  double tol = 1e-4;
  /// The reference of the change test (3.9).
  GalerkinPhiOld phi_old = GalerkinPhiOld::AUTO;
  /// Factors of the norms (sampling::Weighting): energy and space shape the
  /// basis inner product, the residual norms and the scalar-flux norm;
  /// angle scales only the residual norms (the greedy ranking and the
  /// residual test).
  /// NOTE: all factors default to true, like the cross. Guo & Peng weight
  /// every direction equally: set weighting.angle = false for runs that are
  /// to be consistent with the paper.
  sampling::Weighting weighting;
  /// MGS2-A (Alg. 3.2): orthogonality tolerance |X(:, 1)^T psi| <= mgs_eps,
  /// and the most projections per snapshot (at least two; the paper uses
  /// at most 6 and 1e-10).
  double mgs_eps = 1e-10;
  int64_t mgs_max_projections = 6;
  /// A snapshot whose remainder after MGS2-A is at most basis_rtol times its
  /// norm is already in the span: its coordinates are kept but no vector is
  /// appended (also once the basis spans the whole space). An addition to
  /// Alg. 3.2, which always appends: the remainder would be rounding noise.
  double basis_rtol = 1e-12;
  /// Greedy iterations per run (0: until every direction is swept, which
  /// makes the run exact).
  int64_t max_iter = 0;
  /// Seed of the direction sampling (the generator persists across runs).
  uint64_t seed = 0;
  /// Keep the low-rank flux after each transport solve (the solver's
  /// angular flux).
  bool keep_angular_flux = false;
};

} // namespace ttnte::galerkin
