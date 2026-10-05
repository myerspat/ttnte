#pragma once

#include "ttnte/linalg/amen/amen_config.hpp"
#include "ttnte/linalg/format_type.hpp"
#include "ttnte/solvers/enrichment_policy.hpp"
#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::solvers {

/// @brief The AMEn local solver. Dispatches to either ttnte's own native
/// AMEn implementation (`AMEnBackend::NATIVE`, the default) or the vendored
/// torchTT implementation (`AMEnBackend::TORCHTT`), see `backend_`.
class AMEnSolver : public LocalSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<AMEnSolver>;

protected:
  // =================================================================
  // Protected data
  /// Number of sweeps
  int nswp_;
  /// Relative residual.
  double eps_;
  /// Minimum allowed truncation tolerance.
  double eps_floor_;
  /// Forcing for truncation tolerance in an inexact solver.
  double eps_forcing_;
  /// Upper cap on the forced truncation tolerance: eps_ never exceeds this,
  /// however large eps_forcing_ * min_error_ is. min_error_ starts at 1, so
  /// without a cap a forcing factor above about 1 asks the first solves for a
  /// relative truncation tolerance near or above 1, which rounds the solution
  /// away to rank ~1. Defaults to +infinity (no cap, previous behavior).
  double eps_max_;
  /// Maximum allowed rank.
  int max_rank_;
  /// The largest size before switching from a direct solver to GMRES.
  int max_full_;
  /// The rank enrichment size.
  int kickrank_;
  /// ALS enrichment size.
  int kick2_;
  /// Number of GMRES iterations for each subproblem.
  int local_iterations_;
  /// The number of restarts in GMRES.
  int resets_;
  /// Show output.
  bool verbose_;
  /// What preconditioner to use.
  linalg::AMEnPreconditioner preconditioner_;
  /// Which AMEn implementation to dispatch to.
  linalg::AMEnBackend backend_;
  /// Tuning knobs specific to `AMEnBackend::NATIVE`.
  linalg::AMEnNativeOptions native_opts_;
  /// Decides, once per outer DD iteration, whether enrichment should be
  /// active for the next solve() call; null (default) means always enrich,
  /// i.e. no rank-freeze behavior at all.
  EnrichmentPolicy::Ptr enrichment_policy_;
  /// The user-configured kickrank/kick2/als_residual_rank, preserved so
  /// `enrichment_policy_` can restore them after zeroing them out -- unlike
  /// the old sticky-only rank_freeze_eps, a policy may re-enable enrichment.
  int base_kickrank_;
  int base_kick2_;
  int64_t base_als_residual_rank_;
  /// Cached result of `enrichment_policy_`'s last decision (diagnostic --
  /// `should_enrich` itself is stateful/advances internal counters, so it
  /// must not be called again just to query the current state).
  bool rank_frozen_ = false;

  // =================================================================
  // Protected constructors
  AMEnSolver(int nswp = 22, double eps = 1e-10, double eps_forcing = 0.01,
    int max_rank = std::numeric_limits<int>::max(), int max_full = 500,
    int kickrank = 4, int kick2 = 0, int local_iterations = 40, int resets = 2,
    bool verbose = false,
    linalg::AMEnPreconditioner preconditioner =
      linalg::AMEnPreconditioner::NONE,
    linalg::AMEnBackend backend = linalg::AMEnBackend::NATIVE,
    linalg::AMEnNativeOptions native_opts = linalg::AMEnNativeOptions {},
    EnrichmentPolicy::Ptr enrichment_policy = nullptr,
    bool preserve_moments = false, double moment_remainder_relaxation = 1.0,
    double moment_eps = -1.0, int64_t moment_max_rank = -1,
    double eps_max = std::numeric_limits<double>::infinity())
    : nswp_(nswp), eps_(eps), eps_max_(eps_max), max_rank_(max_rank),
      max_full_(max_full), kickrank_(kickrank), kick2_(kick2),
      local_iterations_(local_iterations), resets_(resets), verbose_(verbose),
      preconditioner_(preconditioner), backend_(backend),
      native_opts_(native_opts),
      enrichment_policy_(std::move(enrichment_policy)),
      base_kickrank_(kickrank), base_kick2_(kick2),
      base_als_residual_rank_(native_opts.als_residual_rank)
  {
    // Moment-preserving rounding (see LocalSolver::round_conserved()) --
    // opt-in, forwarded into the base class fields shared by presolve()'s
    // RHS round and this solver's own solution round.
    preserve_moments_ = preserve_moments;
    moment_remainder_relaxation_ = moment_remainder_relaxation;
    moment_eps_ = moment_eps;
    moment_max_rank_ = moment_max_rank;

    if (nswp_ < 1 || eps_ < 0 || max_rank_ < 1 || max_full < 0 ||
        kickrank < 0 || kick2 < 0 || local_iterations_ < 1 || resets_ < 1) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`nswp`, `max_rank`, `local_iterations`, and `resets` must be greater\n"
        "than or equal to 1 and `eps`, `max_full`, `kickrank`, and `kick2`\n"
        "must be greater than or equal to 0");
    }

    if (preconditioner == linalg::AMEnPreconditioner::RANK1 &&
        backend != linalg::AMEnBackend::NATIVE) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`AMEnPreconditioner::RANK1` is only supported with "
        "`AMEnBackend::NATIVE` -- the torchTT backend has no rank-1 "
        "preconditioner");
    }

    if (enrichment_policy_ && backend != linalg::AMEnBackend::NATIVE) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`enrichment_policy` is only supported with `AMEnBackend::NATIVE` -- "
        "the torchTT backend has no zero-enrichment code path");
    }

    if (!(eps_max_ >= eps)) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`eps_max` must be greater than or equal to `eps` -- it caps the "
        "forced truncation tolerance from above, while `eps` is its floor");
    }

    if (moment_remainder_relaxation_ <= 0) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`moment_remainder_relaxation` must be strictly positive -- it "
        "multiplies `eps` to form the moment remainder's truncation "
        "tolerance");
    }

    eps_floor_ = eps;
    eps_forcing_ = eps_forcing;
  }

public:
  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new AMEn solver instance.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new AMEnSolver(std::forward<Args>(args)...));
  }

  /// @brief Solve the local linear system.
  /// @param local_system The local linear system to be solved.
  void solve(const linalg::LinearSystem::Ptr& local_system) override final;

  /// @brief Update min_error_ (via LocalSolver), then force eps_ toward
  /// eps_floor_ as min_error_ improves: eps_ = max(eps_floor_, min(eps_max_,
  /// eps_forcing_ * min_error_)). If an `enrichment_policy_` is set, asks it
  /// (given eps_, error, and rank_metric) whether enrichment should be active
  /// for the next solve() call and toggles kickrank_/kick2_/
  /// native_opts_.als_residual_rank between 0 and their original (base_*)
  /// values accordingly -- a policy may re-enable enrichment later, unlike
  /// the old sticky-only rank_freeze_eps. Since this runs strictly after the
  /// solve() call whose error triggered it, that solve already completed
  /// under the previous decision; a change here only affects solve() calls
  /// from this point on.
  void update_convergence_criteria(
    double error, double rank_metric = 0.0) override
  {
    LocalSolver::update_convergence_criteria(error, rank_metric);
    eps_ = std::max(eps_floor_, std::min(eps_max_, eps_forcing_ * min_error_));

    if (enrichment_policy_) {
      bool enrich = enrichment_policy_->should_enrich(eps_, error, rank_metric);
      kickrank_ = enrich ? base_kickrank_ : 0;
      kick2_ = enrich ? base_kick2_ : 0;
      native_opts_.als_residual_rank = enrich ? base_als_residual_rank_ : 0;
      rank_frozen_ = !enrich;
    }
  }

  // =================================================================
  // Public getters / setters
  /// @return The current truncation tolerance of the solver.
  double get_eps() const override final { return eps_; }
  /// @return The upper cap on the forced truncation tolerance.
  double get_eps_max() const noexcept { return eps_max_; }
  /// @return The maximum rank.
  int64_t get_max_rank() const final override
  {
    return static_cast<int64_t>(max_rank_);
  }
  /// @return Whether `enrichment_policy_` disabled enrichment as of the last
  /// `update_convergence_criteria()` call.
  bool is_rank_frozen() const noexcept { return rank_frozen_; }
  /// @return The enrichment-scheduling policy, or nullptr if enrichment is
  /// always active.
  const EnrichmentPolicy::Ptr& get_enrichment_policy() const noexcept
  {
    return enrichment_policy_;
  }

  /// @return Always FormatType::TENSOR_TRAIN.
  linalg::FormatType get_state_format() override final
  {
    return linalg::FormatType::TENSOR_TRAIN;
  }
};

} // namespace ttnte::solvers
