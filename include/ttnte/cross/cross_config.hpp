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
  /// Inside the sweep solver it is set per transport solve instead (see
  /// SweepConfig::cross_inner_factor and cross_relax).
  double tol = 1e-4;
  /// Approximate the L2-weighted angular flux sqrt(w_d) (R psi)/sqrt(du_g),
  /// M = R^T R the patch mass matrices and du the lethargy widths, so that
  /// pivots, truncation and the hold-out error are L2 quantities. The
  /// lethargy widths are the problem's (CrossProblem::lethargy_widths); in
  /// the sweep solver they come from the cross-section server's group
  /// boundaries.
  bool weighted = true;
  /// Keep the TT of the angular flux after each transport solve.
  bool keep_angular_flux = false;
  /// Index selection adds blocks until the chosen rows have sigma_min /
  /// sigma_max >= select_cond, which bounds the amplification of the
  /// least-squares fits (in (0, 1)).
  double select_cond = 1e-2;
  /// Among the blocks scoring at least select_near times the best, the
  /// selection takes the one needing the fewest new sweeps (in (0, 1]; 1:
  /// always the best score).
  double select_near = 0.25;
  /// Hold-out fibers swept per check.
  int64_t holdout = 16;
  /// Hold-out sample sizes in units of holdout: the pool of swept fibers
  /// judged once (nearly) every pair is swept (resampled_pool), and the
  /// sample of swept fibers whose errors estimate the in-sample stratum of
  /// the global error (swept_sample). Both cost TT evaluations only, >= 1.
  int64_t resampled_pool = 4;
  int64_t swept_sample = 2;
  /// Worst pool pairs forced into the index sets after a failed check, >= 1.
  int64_t forced_per_check = 4;
  /// Acceptance of the hold-out error: holdout_factor * tol (fiber errors
  /// overstate moment errors).
  double holdout_factor = 10.0;
  /// A pass is accepted when the hold-out pool's error, relative to its own
  /// norm, is within holdout_factor * tol. If it is not, and this is set,
  /// the pass may still be accepted on the estimated global relative error
  /// ||A - A~|| / ||A|| (the pool estimating the unswept fibers, a sample of
  /// swept fibers the swept ones), the norm the truncation and the moment
  /// bound use: the pool alone stalls when the unswept fibers are much
  /// smaller than the swept ones. The estimate is formed only then
  /// (SweepRecord::global_error NaN otherwise).
  bool global_holdout = true;
  /// Maximum forward/backward sweeps of the cross per transport solve.
  int64_t max_sweeps = 8;
  /// Stop a run, unconverged, after stall_passes consecutive passes that
  /// leave every rank unchanged and lower the hold-out error by less than
  /// the fraction stall_improvement (0: never). On Big Ten (fiber layout)
  /// runs in that state almost never converged in the passes left.
  int64_t stall_passes = 2;
  double stall_improvement = 0.1;
  /// Cap on every TT rank (<= 0: none).
  int64_t max_rank = 0;
  /// Seed of the hold-out sampling.
  uint64_t seed = 0;
  /// Take the truncated SVDs of tall or wide superblocks through their Gram
  /// matrices (linalg::gram_truncated_svd) instead of a Householder QR:
  /// faster, but singular values only to about 1e-8 relative, so it falls
  /// back to the QR path when the truncation threshold is below gram_floor.
  bool gram_svd = true;
  /// Smallest relative truncation threshold (delta / ||A||_F) taken through
  /// the Gram matrix; below it the QR path (singular values to ~1e-8).
  double gram_floor = 1e-7;
  /// Forward-pass truncation, as a fraction of the per-bond share of tol
  /// (the backward pass uses the full share): the forward cores are refitted
  /// in the backward pass, so they keep more rank. In (0, 1].
  double forward_truncation_share = 0.1;
  /// Smallest sigma_min / sigma_max of a basis inverted through its Gram
  /// matrix (Cholesky, relative error about eps / ratio^2); below it the
  /// pseudo-inverse. Selection aims far above it (select_cond). In (0, 1).
  double gram_solve_min_ratio = 1e-4;
  /// Residual bond growth after the forward pass: the fewest blocks not yet
  /// in the last core's right set holding this share of its residual energy
  /// are added to the set, at most growth_max_blocks or half the set's
  /// blocks, whichever is larger. Share in (0, 1], blocks >= 1.
  double growth_share = 0.5;
  int64_t growth_max_blocks = 4;
  /// Rows of the first Gaussian sketch choosing the initial energies
  /// (doubled until the pivots settle), >= 1.
  int64_t sketch_rows = 32;
  /// Start each run from the right index sets of the previous run instead of
  /// the initial directions. Successive transport solves of the sweep solver
  /// (GMRES products, outer iterations) differ little, so the previous pivots
  /// give the first sweep the previous ranks. Every fiber is still swept anew
  /// and the hold-out check is unchanged. The seeded sets are kept, so they
  /// only grow from run to run: on the sweep-cross regressions a warm start
  /// sweeps 4-14 % more fibers than a cold one.
  // TODO: bound the growth without costing passes (reselecting sets above 4
  // times the rank saved fibers in 1-D/2-D but cost 3-D a pass).
  bool warm_start = false;
};

} // namespace ttnte::cross
