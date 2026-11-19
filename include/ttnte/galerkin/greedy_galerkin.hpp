#pragma once

#include "ttnte/galerkin/galerkin_config.hpp"
#include "ttnte/galerkin/reduced_operator.hpp"
#include "ttnte/sampling/fiber_store.hpp"
#include "ttnte/sampling/sampled_transport.hpp"
#include "ttnte/sampling/sampling_problem.hpp"
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace ttnte::galerkin {

/// @brief One transport solve of a group set by residual-greedy Galerkin
/// sampling of directions (Guo & Peng, 2026, Alg. 3.1-3.2).
///
/// A run starts from an empty basis and repeats:
///  1. sweep p directions over every group of the set (FiberStore), each a
///     snapshot [e_g R_P psi_{d,g}]_g weighted in space and energy;
///  2. orthonormalize the snapshots into the weighted basis X~ by MGS2-A
///     (Alg. 3.2), keeping each swept direction's coordinates (Remark
///     3.3), and border the reduced operator with the unweighted vectors;
///  3. draw q random unswept candidates, solve their reduced systems and
///     score their residuals (FiberStore::residual_norms());
///  4. if max_j ||r_j||* <= tol RMS_j ||b_j||* (3.7), solve every unswept
///     direction and form the scalar flux; stop if it changed by at most
///     tol relative to phi_old (3.9), else make it phi_old. phi_old is the
///     previous run's flux (PREVIOUS_RUN, Guo & Peng under source
///     iteration) or none at a run's start (WITHIN_RUN, Han et al. under
///     GMRES);
///  5. sweep the p candidates with the largest residuals.
/// Swept directions keep their exact fibers (their MGS coordinates); the
/// rest are the reduced solutions X c_d.
///
/// With GalerkinConfig::deim (Han et al. 2026, Sec. 3.2 and 4.2) the runs
/// switch, once a run changes the scalar flux by at most deim.switch_tol,
/// to DEIM source iterations: a run sweeps only the DEIM units I
/// (directions, or (direction, group) pairs with sample = PAIR) and
/// interpolates the rest (the held flux is then the swept fibers with the
/// interpolation coefficients V V(I, :)^{-1}), and a run whose change is
/// within deim.inner_tol updates the angle instead: X = orth(Psi_I), every
/// unit solved with its scattering (ReducedOperator::angular_galerkin()), V
/// and I from that solution. Only such updates are settled() (within the
/// solver's stop tolerance of the previous outer iterate).
class GreedyGalerkin : public sampling::SampledTransport {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<GreedyGalerkin>;

  /// @brief What a run did.
  enum class RunStage : uint8_t {
    /// The greedy sampling (Guo & Peng).
    GREEDY,
    /// A DEIM inner source iteration: the directions I swept, the rest
    /// interpolated.
    DEIM_INNER,
    /// A DEIM inner iteration followed by the angular update.
    DEIM_UPDATE,
  };

  /// @brief One greedy iteration at its residual test: the rank and fibers
  /// scored (before the iteration's own sweep), the residual test's ratio
  /// max_j ||r_j||* / RMS_j ||b_j||* over its candidates (NaN when every
  /// direction is swept), and the relative scalar-flux change of its full
  /// solve (NaN when the residual test failed, or when there was no phi_old
  /// to compare with: the first full solve of a WITHIN_RUN run).
  struct IterationRecord {
    int64_t rank = 0;
    int64_t fibers = 0;
    double residual = 0.0;
    double phi_change = 0.0;
  };

  /// @brief One run (transport solve): the common record, its greedy
  /// iterations, and the snapshots it did not append (basis_rtol, a full
  /// basis) or appended with an orthogonality failure (Alg. 3.2, line 18).
  struct RunRecord : sampling::RunRecord {
    std::vector<IterationRecord> history;
    int64_t dropped_snapshots = 0;
    int64_t orthogonality_failures = 0;
    /// Rank after the truncated SVD (GalerkinConfig::svd_tol; -1: off).
    int64_t truncated_rank = -1;
    /// Basis vectors the run started from (GalerkinConfig::carry_basis;
    /// 0: an empty start).
    int64_t carried_rank = 0;
    /// GalerkinConfig::deim: the run's stage; the relative change of its
    /// weighted scalar flux from the previous run's (NaN: none, or deim
    /// off); the DEIM units it swept (directions, or pairs; 0 in the greedy
    /// stage) and the
    /// quadrature-weighted Lebesgue constant of the selection it used;
    /// after an update, the change from the previous outer iterate (the
    /// previous update, or the greedy iterate of the switch), the solver's
    /// stop tolerance it was tested against (Han et al.'s eps_phi) and
    /// whether the run is settled.
    RunStage stage = RunStage::GREEDY;
    double run_change = 0.0;
    int64_t deim_directions = 0;
    double lebesgue = 0.0;
    double outer_change = 0.0;
    double stop_tolerance = 0.0;
    bool settled = true;
    /// The inner iteration's change grew: the update came early (our
    /// safeguard, not part of Han et al.).
    bool inner_diverged = false;
  };

protected:
  // =================================================================
  // Protected data
  GalerkinConfig config_;
  ReducedOperator::Ptr op_;
  int64_t num_directions_ = 0;
  int64_t num_groups_ = 0;
  int64_t num_space_ = 0;
  /// Sampling units (directions, or (direction, group) pairs u = d G + g,
  /// GalerkinConfig::sample) and the groups of a basis vector (G, or 1).
  int64_t num_units_ = 0;
  int64_t basis_groups_ = 0;
  /// (N_Omega) quadrature weights and (N_Omega, M) harmonics.
  torch::Tensor weights_;
  torch::Tensor harmonics_;
  /// Fixed weights: (num_units) the weight of each unit's coefficients in
  /// the fibers' norm, a_d (directions; X~ carries e_g) or a_d e_g (pairs;
  /// X~ is spatial); (G) e_g; (N_Omega) w_d Y_0(d).
  torch::Tensor unit_weights_;
  torch::Tensor energy_weights_;
  torch::Tensor scalar_weights_;
  std::vector<int64_t> initial_directions_;
  std::mt19937_64 rng_;

  /// The run's weighted orthonormal basis X~, rows of a (capacity, G Ns)
  /// buffer (doubling), rank_ of them in use.
  torch::Tensor basis_rows_;
  int64_t rank_ = 0;
  /// Per unit whether it is swept, and a swept unit's MGS coordinates in
  /// X~ (as many as the rank when it was added).
  std::vector<uint8_t> swept_;
  std::vector<std::vector<double>> coordinates_;
  /// The held flux: (num_units, r) coefficients of every unit on the
  /// unweighted basis (r, basis_groups, Ns) of the reduced operator.
  torch::Tensor coefficients_;
  torch::Tensor basis_;
  /// The untruncated fill's moments, when the held flux was truncated
  /// (moments() returns them).
  torch::Tensor moments_;
  /// The last run's rank after the truncated SVD (-1: svd_tol off).
  int64_t truncated_rank_ = -1;
  /// carry_basis: what the last run leaves for the next, uncut: its
  /// angle-weighted coefficients a_d C (N_Omega, r) and its basis, weighted
  /// (rows of X~) and unweighted (the reduced operator's), (r, G Ns) each;
  /// and the rank the last run started from.
  torch::Tensor carry_coefficients_;
  torch::Tensor carry_rows_;
  torch::Tensor carry_unweighted_;
  int64_t carried_rank_ = 0;
  /// The last full solve's weighted scalar flux (G Ns), phi_old of (3.9);
  /// carried across runs under PREVIOUS_RUN (phi^(n-1) of the paper's
  /// first comparison), reset at each run under WITHIN_RUN.
  torch::Tensor phi_old_;
  /// Per unit its right-hand side's norm ||b_u||* in the current run
  /// (NaN: not computed yet), the residual test's pooled scale.
  std::vector<double> rhs_norm_;

  /// GalerkinConfig::deim: whether the runs are in the DEIM stage; the DEIM
  /// directions I, the interpolation coefficients V V(I, :)^{-1} (N_Omega,
  /// |I|) and the selection's weighted Lebesgue constant; the weighted
  /// scalar flux of the last run and of the last angular update; the last
  /// run's stage, change, outer change and settledness; updates so far.
  bool deim_ = false;
  std::vector<int64_t> deim_rows_;
  torch::Tensor deim_interp_;
  double lebesgue_ = std::numeric_limits<double>::quiet_NaN();
  /// The Lebesgue constant of the selection the last run interpolated with
  /// (NaN for a greedy run), and the switch's flux norm (divergence test).
  double run_lebesgue_ = std::numeric_limits<double>::quiet_NaN();
  double switch_norm_ = 0.0;
  torch::Tensor last_flux_;
  torch::Tensor update_flux_;
  RunStage run_stage_ = RunStage::GREEDY;
  double run_change_ = std::numeric_limits<double>::quiet_NaN();
  double outer_change_ = std::numeric_limits<double>::quiet_NaN();
  bool settled_ = true;
  /// The solver's stop tolerance (set_stop_tolerance(); <= 0: the run's).
  double stop_tol_ = 0.0;
  int64_t updates_ = 0;
  /// The last inner change (infinite after an update or the switch), and
  /// whether the last run's grew from it.
  double inner_change_ = std::numeric_limits<double>::infinity();
  bool inner_diverged_ = false;

  /// Over every run, and in the last run.
  int64_t orthogonality_failures_ = 0;
  int64_t dropped_ = 0;
  int64_t run_failures_ = 0;
  int64_t run_dropped_ = 0;
  std::vector<IterationRecord> history_;

  // =================================================================
  // Protected constructors
  /// @param problem Sizes, weights, harmonics, layout, weighting data and
  /// oracle of the group set (as for the cross).
  /// @param op The set's transport operator, projected per direction.
  /// @param config Options; phi_old must be resolved (not AUTO).
  /// @throws std::runtime_error On inconsistent sizes or options.
  GreedyGalerkin(sampling::SamplingProblem problem, ReducedOperator::Ptr op,
    GalerkinConfig config);

  [[nodiscard]] static std::string ctx(const std::string& f)
  {
    return "ttnte::galerkin::GreedyGalerkin::" + f;
  }

  /// @brief Sweep `units` (a direction over every group, or a pair),
  /// orthonormalize their snapshots into X~ and border the reduced
  /// operator.
  void sample(const std::vector<int64_t>& units, const char* phase);
  /// @brief MGS2-A (linalg::mgs2a()) of one weighted snapshot (G Ns)
  /// against X~: append the new unit vector (unless the snapshot is zero,
  /// in the span to basis_rtol, or the basis full), count drops and
  /// orthogonality failures, and return the snapshot's coordinates (a
  /// column of R in (3.3)).
  std::vector<double> orthonormalize(torch::Tensor snapshot);
  /// @brief Every direction's coefficients: swept ones from their
  /// coordinates, the others solved (coefficients_), and the weighted
  /// scalar flux they give.
  torch::Tensor fill();
  /// @brief The SVD of angle-weighted coefficients a_d C and the rank cut
  /// at svd_tol in svd_norm (svd_tol = 0: every singular value).
  /// @return (P, s, Vh, k).
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> svd_cut(
    const torch::Tensor& Cw) const;
  /// @brief The truncated SVD of the held coefficients (svd_tol): record
  /// the rank, and with `replace` the held flux becomes the truncated one
  /// (the moments kept from before).
  void truncate(bool replace);
  /// @brief carry_basis: keep this run's a_d C and basis (weighted and
  /// unweighted) for the next run. Called at the end of run(), before the
  /// flux is released.
  void keep_basis();
  /// @brief carry_basis: start from the kept basis, rotated by the right
  /// singular vectors of a_d C and cut at this run's tolerance (rows of X~,
  /// and one extend() of the reduced operator).
  void start_from_kept_basis();
  /// @brief Reset the per-run state (empty basis, nothing swept, no held
  /// flux).
  void reset();
  /// @brief The greedy stage's run (Guo & Peng, Alg. 3.1).
  /// @return The run's weighted scalar flux (its final full solve).
  torch::Tensor run_greedy();
  /// @return The units not swept in this run, in order.
  std::vector<int64_t> unswept() const;
  /// @return The store's fibers of `units`: a direction's every group, or
  /// a pair itself.
  std::vector<int64_t> fiber_indices(const std::vector<int64_t>& units) const;
  /// @brief A DEIM-stage run: sweep I, interpolate, and update the angle
  /// once the inner iteration converged.
  void run_deim();
  /// @brief Enter the DEIM stage from the held (greedy) flux: V the left
  /// singular vectors of a_d C, cut at the run's tolerance.
  void enter_deim();
  /// @brief Select the DEIM units of an angular basis (num_units, k):
  /// quadrature-weighted orthonormalized (eq. 3.5, deim_weights()), then
  /// DEIM or Q-DEIM.
  void select_directions(const torch::Tensor& V);
  /// @brief The angular modes of coefficients C (num_units, r): the left
  /// singular vectors of a C (a = unit_weights_), cut at eps_SVD with `cut`
  /// (svd_cut()), and without numerically zero modes (DEIM_ZERO_MODE_TOL,
  /// at least max(m, n) eps of the largest).
  /// @return (num_units, k) V with the weights divided out, and k (0: C is
  /// numerically zero).
  std::pair<torch::Tensor, int64_t> angular_modes(
    const torch::Tensor& C, bool cut) const;
  /// @return (|units|, G, Ns) the stored fibers of directions, or (|units|,
  /// 1, Ns) of pairs, unweighted.
  torch::Tensor unweighted_fibers(const std::vector<int64_t>& units) const;
  /// @return (num_units) the weights of eq. 3.5's orthonormality: w_d, or
  /// w_d e_g^2 for pairs (ours).
  torch::Tensor deim_weights() const;
  /// @return (G Ns) the held flux's scalar flux in the basis' weighting
  /// (space R_P and energy e_g), as fill() returns it.
  torch::Tensor weighted_scalar_flux() const;
  /// @return Up to n distinct unswept directions, uniformly at random.
  std::vector<int64_t> draw(int64_t n);
  /// @return Whether a unit is a (direction, group) pair.
  bool pairs() const noexcept { return config_.sample == GalerkinSample::PAIR; }

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new GreedyGalerkin(std::forward<Args>(args)...));
  }

  void run() override final;
  void restart() override final;
  void set_tolerance(double tol) override final { config_.tol = tol; }
  torch::Tensor moments() const override final;
  torch::Tensor evaluate(const std::vector<int64_t>& directions,
    const std::vector<int64_t>& groups,
    const std::vector<int64_t>& space) const override final;
  void release() override final;
  /// @brief With keep_angular_flux and svd_tol: the held flux becomes the
  /// truncated one (after the solver read the moments and traces).
  void compress() override final;
  bool settled() const noexcept override final { return settled_; }
  void set_stop_tolerance(double tol) override final { stop_tol_ = tol; }

  // =================================================================
  // Public getters
  double tolerance() const noexcept override final { return config_.tol; }
  bool holds_flux() const noexcept override final
  {
    return coefficients_.defined();
  }
  bool keeps_flux() const noexcept override final
  {
    return config_.keep_angular_flux;
  }
  std::vector<int64_t> ranks() const override final { return {rank_}; }
  // error() (the base's): a greedy run's last residual test ratio max_j
  // ||r_j||* / RMS_j ||b_j||* (0 when the run ended with every direction
  // swept); a DEIM run's inner change.
  /// @return The options in effect (phi_old resolved, tol the last set).
  const GalerkinConfig& config() const noexcept { return config_; }
  const std::vector<IterationRecord>& history() const noexcept
  {
    return history_;
  }
  sampling::RunRecord::Ptr record() const override final;
  /// @return Nonzero snapshots not appended (in the span to basis_rtol, or
  /// the basis full), over every run.
  int64_t dropped_snapshots() const noexcept { return dropped_; }
  /// @return Snapshots whose MGS2-A cleanups ended above mgs_eps (Alg. 3.2,
  /// line 18), over every run.
  int64_t orthogonality_failures() const noexcept
  {
    return orthogonality_failures_;
  }
  /// @return Whether each unit (direction, or pair d G + g) was swept in
  /// the last run.
  const std::vector<uint8_t>& swept() const noexcept { return swept_; }
  /// @return The last run's rank after the truncated SVD (-1: svd_tol off).
  int64_t truncated_rank() const noexcept { return truncated_rank_; }
  /// @return Basis vectors the last run started from (carry_basis).
  int64_t carried_rank() const noexcept { return carried_rank_; }
  /// @return (G Ns) the weighted scalar flux (3.9) compares the next full
  /// solve with (phi^(n-1): the last full solve's), or undefined if none yet.
  const torch::Tensor& phi_old() const noexcept { return phi_old_; }
  /// @return Whether the runs are in the DEIM stage (GalerkinConfig::deim).
  bool in_deim_stage() const noexcept { return deim_; }
  /// @return The DEIM directions of the next run.
  const std::vector<int64_t>& deim_directions() const noexcept
  {
    return deim_rows_;
  }
  /// @return Angular updates so far.
  int64_t num_updates() const noexcept { return updates_; }
};

} // namespace ttnte::galerkin
