#pragma once

#include "ttnte/sampling/sampling_problem.hpp"
#include "ttnte/sampling/weighting.hpp"
#include <c10/core/Allocator.h>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <vector>

namespace ttnte::sampling {

/// @brief The swept fibers of one sampled transport solve, weighted as they
/// finish.
///
/// Every request goes to the problem's sweep oracle once, for the pairs not
/// yet stored; directions sharing a group list are swept as one block. A
/// finished fiber of (direction d, group g) is stored as
///   a_d e_g R_P psi_{d,g}
/// on each patch P, each factor only if enabled (see Weighting). The
/// weighting runs inside the oracle's tasks as fibers finish.
///
/// Shared by the samplers of one transport solve (LayoutCross and the
/// Galerkin samplers): they differ in which pairs they request and what they
/// fit, not in how pairs are swept, weighted and stored.
class FiberStore {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<FiberStore>;

  /// @brief One request to the sweep oracle: the phase of the sampler that
  /// made it, its new fibers, sweep blocks (a direction list sharing one
  /// group list) and distinct directions, and its seconds.
  struct SweepCall {
    std::string phase;
    int64_t fibers = 0;
    int64_t blocks = 0;
    int64_t directions = 0;
    double seconds = 0.0;
  };

protected:
  // =================================================================
  // Protected data
  /// The problem (sizes, oracle, weights, patch layout); not owned elsewhere.
  SamplingProblem problem_;
  int64_t num_directions_ = 0;
  int64_t num_groups_ = 0;
  int64_t num_space_ = 0;
  /// The factors in effect: those asked for that the problem has data for
  /// (none until the constructor enables them).
  Weighting weighting_ {false, false, false};

  /// Weight factors: sqrt(w_d) per direction and 1 / sqrt(du_g) per group
  /// (1 where disabled), and the Cholesky factors of the patch mass matrices
  /// (empty where disabled), banded and, for small patches, dense.
  std::vector<double> angle_weight_;
  std::vector<double> energy_weight_;
  std::vector<torch::Tensor> mass_factors_;
  std::vector<int64_t> mass_kd_;
  /// The same factors as dense upper triangular R_P for patches small enough
  /// to apply them to batches of fibers with BLAS-3 (undefined otherwise).
  std::vector<torch::Tensor> mass_dense_;
  /// The dense factors grouped by size, applied as one batched product or
  /// triangular solve per group (many small patches, e.g. FDG elements,
  /// would otherwise cost one call each): (B, n, n) factors and the (B n)
  /// fiber columns of their patches, patch by patch. Patches without a
  /// dense factor (banded_patches_) are applied one by one.
  struct DenseGroup {
    torch::Tensor factors;
    torch::Tensor columns;
  };
  std::vector<DenseGroup> dense_groups_;
  std::vector<size_t> banded_patches_;

  /// Swept fibers (weighted): one block per sweep request, never moved, and
  /// each pair's fiber within it (null: not swept). Blocks come from
  /// torch's CPU allocator, which puts large ones on huge pages when
  /// THP_MEM_ALLOC_ENABLE=1.
  std::vector<c10::DataPtr> blocks_;
  std::vector<const double*> slot_;
  int64_t num_swept_ = 0;
  /// Sum of squared norms of the stored (weighted) fibers.
  double swept_sq_ = 0.0;

  /// Oracle seconds since the last clear(), and accumulated over every
  /// request: "sweep" (the oracle, including the weighting inside it) and
  /// "weight_cpu" (seconds in the weighting callbacks summed over the
  /// threads running them, overlapping "sweep").
  double sweep_seconds_ = 0.0;
  std::map<std::string, double> timings_;
  /// Every request, over every run.
  std::vector<SweepCall> sweep_calls_;

  // =================================================================
  // Protected constructors
  /// @param weighting The factors to apply; energy and space are applied
  /// only where the problem carries lethargy widths and mass bands.
  /// @throws std::runtime_error If the problem's patch layout is
  /// inconsistent with its sizes or, for an enabled factor, its mass bands
  /// or lethargy widths are, or a mass matrix is not positive definite.
  FiberStore(SamplingProblem problem, Weighting weighting);

  [[nodiscard]] static std::string ctx(const std::string& f)
  {
    return "ttnte::sampling::FiberStore::" + f;
  }

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new FiberStore(std::forward<Args>(args)...));
  }

  /// @brief Make every fiber of `pairs` (d * G + g) available: sweep those
  /// not yet stored in one oracle request (directions sharing a group list
  /// form one block), weight them as they finish, store them in a new block,
  /// and record the request.
  /// @param pairs Pair indices, in any order, repeats allowed.
  /// @param phase The sampler phase making the request (for SweepCall).
  void ensure(std::vector<int64_t> pairs, const char* phase);

  /// @brief Start a new run: drop every stored fiber and zero the run's
  /// counters (fibers, squared norm, sweep seconds); the SweepCall records
  /// and accumulated timings are kept.
  void clear();

  /// @brief Free the stored fibers but keep the run's counters (the
  /// samplers report them after the run).
  void release();

  /// @brief Apply R_P (or R_P^{-1}) of every patch to `count` contiguous
  /// fibers at `rows`, serially (callers parallelize over rows). Nothing
  /// happens without space weighting.
  void apply_mass(double* rows, int64_t count, bool inverse) const;

  /// @brief Apply R_P^{-T} of every patch to `count` contiguous rows at
  /// `rows`, serially: the dual of apply_mass(), for functionals (residuals
  /// tested against the basis), since ||R_P^{-T} r|| = sqrt(r^T M_P^{-1} r).
  /// Nothing happens without space weighting.
  void apply_dual_mass(double* rows, int64_t count) const;

  /// @brief Error-weighted dual norms of transport residuals: per residual
  ///   sqrt(sum_g || a_d e_g R_P^{-T} r_{d,g} ||^2)
  /// with the factors in effect (see weighting()). R_P^{-T} is the dual
  /// (Riesz) norm in space, ||R^{-T} r||^2 = r^T M^{-1} r, since a residual
  /// is a functional; the angle and energy factors multiply, as they weight
  /// the error e = A^{-1} r the residual implies (A_d is block diagonal in
  /// direction and group), so a_d e_g ||R e|| <= kappa a_d e_g ||R^{-T} r||.
  /// This is not the dual of the whole weighted norm, which would divide by
  /// a_d e_g (derivation in the guo-galerkin-cpp report, step 4b). With no
  /// factor in effect it is the Euclidean norm of r (Guo & Peng's l2
  /// residual).
  /// @param directions (D) direction of each residual.
  /// @param residuals (D, num_groups, num_space) residuals b_d - A_d x in the
  /// fiber layout (integrals against the test functions, unweighted).
  /// @return (D) norms.
  /// @throws std::runtime_error On a wrong shape or a direction out of
  /// range.
  torch::Tensor residual_norms(const std::vector<int64_t>& directions,
    const torch::Tensor& residuals) const;

  /// @brief residual_norms() for (direction, group) pairs: one residual of
  /// one group per pair (pair index d num_groups + g), the same weighting
  /// (R_P^{-T} in space, times a_d e_g).
  /// @param pairs (U) pair of each residual.
  /// @param residuals (U, num_space), or (U, 1, num_space).
  /// @return (U) norms.
  /// @throws std::runtime_error On a wrong shape or a pair out of range.
  torch::Tensor pair_residual_norms(
    const std::vector<int64_t>& pairs, const torch::Tensor& residuals) const;

  // =================================================================
  // Public getters
  /// @return Whether fiber `pair` is stored.
  bool swept(int64_t pair) const { return slot_[pair] != nullptr; }
  /// @return The stored (weighted) fiber of `pair`, num_space values; it
  /// must be swept.
  const double* fiber(int64_t pair) const { return slot_[pair]; }
  /// @return The slot of every pair (d * G + g), null where not swept.
  const std::vector<const double*>& slots() const noexcept { return slot_; }
  int64_t num_swept() const noexcept { return num_swept_; }
  /// @return Sum of squared norms of the stored (weighted) fibers.
  double swept_sq() const noexcept { return swept_sq_; }
  /// @return Oracle seconds since the last clear() (the current run).
  double sweep_seconds() const noexcept { return sweep_seconds_; }
  /// @return Accumulated "sweep" and "weight_cpu" seconds (see timings_).
  const std::map<std::string, double>& timings() const noexcept
  {
    return timings_;
  }
  const std::vector<SweepCall>& sweep_calls() const noexcept
  {
    return sweep_calls_;
  }
  const SamplingProblem& problem() const noexcept { return problem_; }
  /// @return sqrt(w_d) per direction (1 without angle weighting).
  const std::vector<double>& angle_weights() const noexcept
  {
    return angle_weight_;
  }
  /// @return 1 / sqrt(du_g) per group (1 without energy weighting).
  const std::vector<double>& energy_weights() const noexcept
  {
    return energy_weight_;
  }
  /// @return The factors in effect (asked for, with the data to apply them).
  const Weighting& weighting() const noexcept { return weighting_; }
  /// @return Whether fibers are weighted by R_P.
  bool has_space_weights() const noexcept { return weighting_.space; }
};

} // namespace ttnte::sampling
