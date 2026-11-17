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

/// @brief What one greedy sample is.
enum class GalerkinSample : uint8_t {
  /// A direction swept over every group of the set; basis vectors span space
  /// x energy, one reduced system per direction (Guo & Peng's multigroup).
  DIRECTION,
  /// A (direction, group) pair: a spatial basis shared by the groups, one
  /// reduced system per pair (ours, a labelled ablation: "Guo + energy
  /// sampling"). The same as DIRECTION with one group.
  PAIR,
};

/// @brief The norm of the discarded singular values in the svd_tol cut,
/// relative to the same norm of all of them.
enum class SvdNorm : uint8_t {
  /// sqrt(sum_{i > r'} s_i^2) <= svd_tol sqrt(sum_i s_i^2): the truncated
  /// flux's weighted L2 error is at most svd_tol (linalg::truncated_svd(),
  /// the cross's rule).
  FROBENIUS,
  /// sum_{i > r'} s_i <= svd_tol sum_i s_i (the nuclear norm): Guo & Peng's
  /// printed rule (with the head sum read as the tail), stricter, for their
  /// reported ranks.
  NUCLEAR,
};

/// @brief How the DEIM stage picks its directions from the angular basis V.
enum class DeimSelect : uint8_t {
  /// Han et al.'s Alg. B.1 (linalg::deim_select): column by column, so it
  /// depends on the order of V's columns.
  DEIM,
  /// Q-DEIM (linalg::qdeim_select): column-pivoted QR of V^T, invariant to
  /// orthogonal changes of V's basis.
  QDEIM,
};

/// @brief Han et al. (2026)'s DEIM-based source iteration (Sec. 3.2) after
/// the greedy stage (the hybrid of their Sec. 4.2). Source iteration only:
/// each run of the sampler is then one source iteration.
///  - Greedy stage: ordinary runs until a run's scalar flux changes from the
///    previous run's by at most switch_tol (and the run's own tolerance is
///    at most switch_tol: under relaxed tolerances loose runs can agree
///    closely while being coarse); the angular basis V is then the
///    left singular vectors of that run's coefficients (Alg. 3.1 step 17,
///    cut at eps_SVD = svd_tol in svd_norm, 0: every nonzero mode;
///    quadrature-weighted orthonormal), and DEIM picks |I| = rank(V)
///    directions I. The stage keeps that rank: its flux lies in
///    span(Psi_I), so the greedy rank at the switch bounds its accuracy
///    (switch late enough, i.e. switch_tol small enough).
///  - DEIM stage, inner iterations (eq. 3.2): a run sweeps only the
///    directions I and interpolates every other direction,
///    psi_d = Psi_I V(I, :)^{-T} V(d, :)^T.
///  - Once a run's flux changes by at most inner_tol (eq. 3.3: the inner
///    source iteration converged; 1e-6 as in Han et al.'s tests), the angular
///    update (eq. 3.4-3.5): X = orth(Psi_I), every direction solved together
///    with its within-set scattering (ProjectedTransport::angular_galerkin), V
///    from that solution and I reselected. As in the paper ("set phi^(n+1) =
///    phi^(n+1,l)") the run's flux stays the interpolant: the update only
///    supplies V and I for the next inner iterations. The run is settled
///    when its flux changed by at most the run's tolerance from the previous
///    update's (Sec. 3.2 (iv)); the sweep solver stops only on settled runs.
/// Two labelled additions (ours, off by default):
///  - galerkin_flux: the update's run holds the Galerkin solution X L
///    instead of the interpolant (the solver's next source and the outer
///    test then use it).
///  - update_on_growth: an inner run whose change exceeds the previous inner
///    run's updates at once. Interpolation amplifies the error outside
///    span(V) by up to the Lebesgue constant, and the interpolated source
///    iteration can diverge with strong scattering; Han et al. have no
///    safeguard. Each run records whether its change grew either way.
struct DeimConfig {
  bool enabled = false;
  /// Switch from the greedy stage when a run changes the scalar flux by at
  /// most this (relative).
  double switch_tol = 1e-4;
  /// Inner convergence (eq. 3.3: relative change of the scalar flux between
  /// runs) that triggers an angular update. Han et al. use 1e-6 (Sec. 6.3).
  /// 0 (ours): the run's tolerance, which the solver relaxes with the outer
  /// change, so a drifting DEIM stage loosens its own inner test.
  double inner_tol = 1e-6;
  DeimSelect select = DeimSelect::DEIM;
  /// Ours (ablation): an update's run holds the Galerkin flux X L, not the
  /// interpolant.
  bool galerkin_flux = false;
  /// Ours (safeguard): a growing inner change triggers the update at once.
  bool update_on_growth = false;
};

/// @brief Options of the residual-greedy Galerkin transport solve (Guo &
/// Peng, 2026, Alg. 3.1-3.2, multigroup by angular augmentation over a group
/// set's groups).
struct GalerkinConfig {
  /// Directions swept per greedy iteration (p).
  int64_t per_iter = 1;
  /// Random unswept candidates scored per iteration (q > p).
  int64_t candidates = 8;
  /// The sampling unit (directions or pairs: per_iter, candidates and the
  /// start count units).
  GalerkinSample sample = GalerkinSample::DIRECTION;
  /// First snapshots.
  GalerkinStart start = GalerkinStart::RANDOM;
  /// Relative accuracy of a run: the residual test (3.7) is
  ///   max_j ||r_j||* <= tol * RMS_j ||b_j||*
  /// over the q candidates (b_j their right-hand sides), and the change test
  /// (3.9) ||phi - phi_old||_w <= tol ||phi||_w. Inside the sweep solver it
  /// is set per transport solve instead (SweepConfig::sampling_inner_factor and
  /// sampling_relax, as for the cross).
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
  /// Truncated SVD after a run (Guo & Peng, "Construction of V(n) and S(n)
  /// after convergence", eps_SVD = 1e-8 in their tests; 0: off). The
  /// coefficients a_d C (a_d = sqrt(w_d) under weighting.angle, else 1) on
  /// the weighted orthonormal basis are cut at the smallest rank r' whose
  /// discarded singular values are at most svd_tol relative, in svd_norm.
  /// Each run records r' (truncated_rank). With keep_angular_flux the held flux
  /// becomes the rank-r' one (coefficients and basis) at compress(), which the
  /// sweep solver calls after it read the moments and the lagged traces: the
  /// moments stay those of the untruncated fill (the paper's phi^(n,*)) and
  /// the iteration does not change, only the angular flux it keeps.
  double svd_tol = 0.0;
  SvdNorm svd_norm = SvdNorm::FROBENIUS;
  /// Carry the basis from one run (transport solve) to the next: NOT part
  /// of Guo & Peng (each of their inner loops starts from p random
  /// directions), so a labelled ablation. A run keeps its angle-weighted
  /// coefficients a_d C and its basis; the next run factors them, a_d C =
  /// U S V^T (linalg::truncated_svd), cut at its own tolerance in the
  /// Frobenius norm (||tail||_F <= tol ||a_d C||_F: the carried flux's
  /// weighted L2 error is at most tol, so the carried rank follows the
  /// accuracy each solve asks for), and starts from the rotated basis V^T
  /// X~ (bordered in one extend(), its sources projected anew) instead of
  /// an empty basis. No direction counts as swept at the start (its fiber
  /// belonged to the previous sources).
  bool carry_basis = false;
  /// Han et al.'s DEIM stage after the greedy one (source iteration only,
  /// direction sampling; svd_tol does not compress DEIM-stage fluxes).
  DeimConfig deim;
};

} // namespace ttnte::galerkin
