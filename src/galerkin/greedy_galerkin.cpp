#include "ttnte/galerkin/greedy_galerkin.hpp"
#include "ttnte/linalg/deim.hpp"
#include "ttnte/linalg/gram_schmidt.hpp"
#include "ttnte/linalg/lapack.hpp"
#include "ttnte/linalg/matrix_ops.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace ttnte::galerkin {

namespace {

// Default; CMake overrides it (TTNTE_DEIM_ZERO_MODE_TOL)
#ifndef TTNTE_DEIM_ZERO_MODE_TOL
#define TTNTE_DEIM_ZERO_MODE_TOL 1e-14
#endif

/// Singular values of the greedy coefficients at most this times the largest
/// are numerically zero: enter_deim() drops their modes (rounding noise that
/// DEIM would otherwise spend directions on). Not an accuracy knob: that is
/// GalerkinConfig::svd_tol.
constexpr double DEIM_ZERO_MODE_TOL = TTNTE_DEIM_ZERO_MODE_TOL;

using Clock = std::chrono::steady_clock;
const auto F64 = torch::TensorOptions().dtype(torch::kFloat64);

double elapsed(Clock::time_point t0)
{
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

torch::Tensor index_tensor(const std::vector<int64_t>& v)
{
  return torch::tensor(v, torch::kInt64);
}

} // namespace

GreedyGalerkin::GreedyGalerkin(sampling::SamplingProblem problem,
  ReducedOperator::Ptr op, GalerkinConfig config)
  : config_(config), op_(std::move(op)), rng_(config.seed)
{
  if (!op_)
    throw utils::runtime_error(ctx("GreedyGalerkin"), "No reduced operator");
  if (config_.per_iter < 1)
    throw utils::runtime_error(ctx("GreedyGalerkin"), "per_iter must be >= 1");
  if (config_.candidates < config_.per_iter)
    throw utils::runtime_error(
      ctx("GreedyGalerkin"), "candidates must be >= per_iter");
  if (!(config_.tol > 0.0))
    throw utils::runtime_error(ctx("GreedyGalerkin"), "tol must be positive");
  if (!(config_.mgs_eps > 0.0))
    throw utils::runtime_error(
      ctx("GreedyGalerkin"), "mgs_eps must be positive");
  if (config_.mgs_max_projections < 2)
    throw utils::runtime_error(
      ctx("GreedyGalerkin"), "mgs_max_projections must be >= 2");
  if (!(config_.basis_rtol >= 0.0))
    throw utils::runtime_error(
      ctx("GreedyGalerkin"), "basis_rtol must be >= 0");
  if (!(config_.svd_tol >= 0.0 && config_.svd_tol < 1.0))
    throw utils::runtime_error(
      ctx("GreedyGalerkin"), "svd_tol must be in [0, 1)");
  if (config_.max_iter < 0)
    throw utils::runtime_error(ctx("GreedyGalerkin"), "max_iter must be >= 0");
  if (config_.deim.enabled && !(config_.deim.switch_tol > 0.0))
    throw utils::runtime_error(
      ctx("GreedyGalerkin"), "deim.switch_tol must be positive");
  if (config_.deim.enabled && !(config_.deim.inner_tol >= 0.0))
    throw utils::runtime_error(
      ctx("GreedyGalerkin"), "deim.inner_tol must be >= 0");
  if (config_.phi_old == GalerkinPhiOld::AUTO)
    throw utils::runtime_error(ctx("GreedyGalerkin"),
      "phi_old must be resolved (PREVIOUS_RUN or WITHIN_RUN) before "
      "construction");
  weights_ = problem.weights.to(torch::kFloat64).contiguous();
  harmonics_ = problem.harmonics.to(torch::kFloat64).contiguous();
  initial_directions_ = problem.initial_directions;
  store_ = sampling::FiberStore::create(std::move(problem), config_.weighting);
  num_directions_ = op_->num_directions();
  num_groups_ = store_->problem().num_groups;
  num_space_ = store_->problem().num_space;
  num_units_ = op_->num_units();
  basis_groups_ = op_->basis_groups();
  if (weights_.size(0) != num_directions_ ||
      harmonics_.size(0) != num_directions_)
    throw utils::runtime_error(ctx("GreedyGalerkin"),
      "The problem's and the operator's directions differ");
  if (num_units_ !=
        (pairs() ? num_directions_ * num_groups_ : num_directions_) ||
      basis_groups_ != (pairs() ? 1 : num_groups_))
    throw utils::runtime_error(ctx("GreedyGalerkin"),
      "The operator's sampling unit differs from the configured sample");
  if (config_.deim.enabled && pairs())
    throw utils::runtime_error(ctx("GreedyGalerkin"),
      "deim needs direction sampling (GalerkinSample::DIRECTION)");
  swept_.assign(static_cast<size_t>(num_units_), 0);
  coordinates_.resize(static_cast<size_t>(num_units_));
}

std::vector<int64_t> GreedyGalerkin::draw(int64_t n)
{
  std::vector<int64_t> pool;
  for (int64_t u = 0; u < num_units_; u++)
    if (!swept_[u])
      pool.push_back(u);
  const int64_t k = std::min<int64_t>(n, static_cast<int64_t>(pool.size()));
  // Partial Fisher-Yates: the first k of a uniform random permutation
  for (int64_t i = 0; i < k; i++) {
    std::uniform_int_distribution<int64_t> pick(
      i, static_cast<int64_t>(pool.size()) - 1);
    std::swap(pool[i], pool[pick(rng_)]);
  }
  pool.resize(static_cast<size_t>(k));
  return pool;
}

torch::Tensor GreedyGalerkin::unit_weights() const
{
  const auto a = torch::tensor(store_->angle_weights(), F64); // (N_Omega)
  if (!pairs())
    return a;
  const auto e = torch::tensor(store_->energy_weights(), F64); // (G)
  return (a.unsqueeze(1) * e.unsqueeze(0)).reshape({-1});      // u = d G + g
}

std::vector<double> GreedyGalerkin::orthonormalize(torch::Tensor snapshot)
{
  // MGS2-A (Alg. 3.2) of one snapshot against the rank_ rows of X~, with
  // our basis_rtol addition: a snapshot already in the span to basis_rtol
  // of its norm (or any once the basis spans the whole space) keeps its
  // coordinates but appends no vector, since its remainder would be
  // rounding noise (past the space's dimension, a dependent vector and a
  // singular reduced matrix)
  const int64_t L = basis_groups_ * num_space_;
  const auto X = basis_rows_.defined() ? basis_rows_.narrow(0, 0, rank_)
                                       : torch::empty({0, L}, F64);
  linalg::MGS2Options options;
  options.eps = config_.mgs_eps;
  options.max_projections = config_.mgs_max_projections;
  options.rtol = config_.basis_rtol;
  const auto result = linalg::mgs2a(X, snapshot, options);
  std::vector<double> out(static_cast<size_t>(rank_));
  std::copy_n(result.coefficients.data_ptr<double>(), rank_, out.begin());
  if (!result.appended) {
    if (result.norm > 0.0)
      dropped_++, run_dropped_++;
    return out;
  }
  if (!result.orthogonal)
    orthogonality_failures_++, run_failures_++;

  // Append the unit remainder as row rank_ (capacity doubles)
  const int64_t cap = basis_rows_.defined() ? basis_rows_.size(0) : 0;
  if (rank_ == cap) {
    auto grown = torch::zeros({std::max<int64_t>(4, 2 * cap), L}, F64);
    if (rank_ > 0)
      grown.narrow(0, 0, rank_).copy_(basis_rows_.narrow(0, 0, rank_));
    basis_rows_ = grown;
  }
  basis_rows_[rank_].copy_(snapshot);
  out.push_back(result.alpha);
  rank_++;
  return out;
}

void GreedyGalerkin::sample(
  const std::vector<int64_t>& units, const char* phase)
{
  const int64_t G = num_groups_, Ns = num_space_, Gb = basis_groups_;
  // The fibers: every group of each direction, or each pair itself (the
  // store's pair index is the unit, d G + g)
  std::vector<int64_t> fibers;
  for (int64_t u : units)
    if (pairs())
      fibers.push_back(u);
    else
      for (int64_t g = 0; g < G; g++)
        fibers.push_back(u * G + g);
  store_->ensure(fibers, phase);

  // Snapshots: the stored fibers a_d e_g R_P psi_{d,g} without the angle
  // factor a_d (and for a pair without e_g too: a spatial basis). The span
  // would not care, but a swept unit's MGS coordinates become its row of C
  // (Remark 3.3), so they must reproduce the basis' own weighting of psi;
  // the removed factors enter only the residual norms. They are the store's
  // own (1 where disabled), so this undoes exactly what the store applied.
  // Orthonormalized in turn (each new vector orthogonalizes the next, as in
  // Alg. 3.2)
  auto t0 = Clock::now();
  const int64_t r0 = rank_;
  const auto w = unit_weights();
  const double* wu = w.data_ptr<double>();
  for (int64_t u : units) {
    auto psi = torch::empty({Gb * Ns}, F64);
    double* p = psi.data_ptr<double>();
    if (pairs())
      std::copy_n(store_->fiber(u), Ns, p);
    else
      for (int64_t g = 0; g < G; g++)
        std::copy_n(store_->fiber(u * G + g), Ns, p + g * Ns);
    psi /= wu[u];
    coordinates_[u] = orthonormalize(psi);
    swept_[u] = 1;
  }
  timings_["orth"] += elapsed(t0);

  // The new rows, unweighted (R_P^{-1}, and 1 / e_g for direction
  // sampling) for the reduced operator
  t0 = Clock::now();
  const int64_t p = rank_ - r0;
  if (p > 0) {
    auto added = basis_rows_.narrow(0, r0, p).clone().view({p, Gb, Ns});
    store_->apply_mass(added.data_ptr<double>(), p * Gb, true);
    if (!pairs()) {
      const auto& energy = store_->energy_weights();
      for (int64_t g = 0; g < G; g++)
        added.select(1, g).div_(energy[g]);
    }
    op_->extend(added);
  }
  timings_["project"] += elapsed(t0);
}

torch::Tensor GreedyGalerkin::fill()
{
  const int64_t U = num_units_, r = rank_;
  auto C = torch::zeros({U, r}, F64);
  if (r > 0) {
    std::vector<int64_t> unswept;
    for (int64_t u = 0; u < U; u++)
      if (!swept_[u])
        unswept.push_back(u);
    if (!unswept.empty())
      C.index_copy_(0, index_tensor(unswept), op_->solve(unswept));
    // Swept units: their exact fibers (Remark 3.3), zero beyond the rank at
    // which they were added (later vectors are orthogonal to them)
    double* c = C.data_ptr<double>();
    for (int64_t u = 0; u < U; u++)
      if (swept_[u])
        std::copy(coordinates_[u].begin(), coordinates_[u].end(), c + u * r);
  }
  coefficients_ = C;
  basis_ = op_->basis();
  if (r == 0)
    return torch::zeros({num_groups_ * num_space_}, F64);
  // Weighted scalar flux, the space and energy factors only (X~ carries no
  // angle factor): X~ a with a = C^T (w Y_0) for directions; per group g
  // e_g X~ a_g, a_g = sum_d w_d Y_0(d) c_{d,g}, for pairs (spatial X~)
  const auto wy = weights_ * harmonics_.select(1, 0);
  const auto X = basis_rows_.narrow(0, 0, r);
  if (!pairs())
    return X.t().mv(C.t().mv(wy));
  const auto A =
    torch::einsum("d,dgr->gr", {wy, C.view({num_directions_, num_groups_, r})});
  const auto e = torch::tensor(store_->energy_weights(), F64).unsqueeze(1);
  return (A.mm(X) * e).reshape({-1});
}

void GreedyGalerkin::reset()
{
  store_->clear();
  op_->reset();
  rank_ = 0;
  std::fill(swept_.begin(), swept_.end(), 0);
  for (auto& c : coordinates_)
    c.clear();
  coefficients_ = torch::Tensor();
  basis_ = torch::Tensor();
  moments_ = torch::Tensor();
  truncated_rank_ = -1;
  history_.clear();
  run_dropped_ = 0;
  run_failures_ = 0;
  carried_rank_ = 0;
  run_change_ = std::numeric_limits<double>::quiet_NaN();
  outer_change_ = std::numeric_limits<double>::quiet_NaN();
}

// One run: the greedy stage, or (GalerkinConfig::deim) a DEIM-stage source
// iteration. After a greedy run, the switch: its flux changed from the
// previous run's by at most deim.switch_tol
void GreedyGalerkin::run()
{
  const auto t0 = Clock::now();
  if (deim_) {
    run_deim();
  } else {
    run_greedy();
    run_stage_ = RunStage::GREEDY;
    settled_ = true;
    if (config_.deim.enabled) {
      auto td = Clock::now();
      const auto phi = weighted_scalar_flux();
      if (last_flux_.defined())
        run_change_ = (phi - last_flux_).norm().item<double>() /
                      std::max(phi.norm().item<double>(),
                        std::numeric_limits<double>::min());
      last_flux_ = phi;
      // Han et al. switch once the source iteration has settled to
      // switch_tol; under relaxed tolerances two loose runs can agree that
      // well while each is far coarser (and the rank, which the DEIM stage
      // keeps, too small), so the run itself must be that accurate as well
      if (run_change_ <= config_.deim.switch_tol &&
          config_.tol <= config_.deim.switch_tol)
        enter_deim();
      timings_["deim"] += elapsed(td);
    }
  }
  seconds_ = elapsed(t0);
}

void GreedyGalerkin::run_greedy()
{
  reset();
  if (config_.carry_basis)
    start_from_kept_basis();
  rhs_norm_.assign(num_units_, std::numeric_limits<double>::quiet_NaN());
  // (3.9)'s reference: the previous run's flux (carried) or none yet
  if (config_.phi_old == GalerkinPhiOld::WITHIN_RUN)
    phi_old_ = torch::Tensor();
  converged_ = false;
  error_ = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Start: p random units, or the cross's initial directions (every group
  // of each for pairs)
  std::vector<int64_t> start;
  if (config_.start == GalerkinStart::INITIAL_DIRECTIONS) {
    for (int64_t d : initial_directions_) {
      if (d < 0 || d >= num_directions_)
        continue;
      for (int64_t g = 0; g < (pairs() ? num_groups_ : 1); g++) {
        const int64_t u = pairs() ? d * num_groups_ + g : d;
        if (std::find(start.begin(), start.end(), u) == start.end())
          start.push_back(u);
      }
    }
  }
  if (start.empty())
    start = draw(config_.per_iter);
  sample(start, "init");

  const int64_t q = config_.candidates;
  bool filled = false;
  for (int64_t iter = 1;; iter++) {
    IterationRecord record {rank_, store_->num_swept(), nan, nan};
    filled = false;

    // Every direction swept: the flux is exact
    auto candidates = draw(q);
    if (candidates.empty()) {
      auto tf = Clock::now();
      fill();
      timings_["fill"] += elapsed(tf);
      filled = true;
      error_ = 0.0;
      converged_ = true;
      history_.push_back(record);
      break;
    }

    // Score the candidates: residuals of their reduced solutions, and their
    // right-hand sides (residuals of zero) for the pooled scale. A
    // direction's right-hand side is fixed within the run (the sources and
    // traces are), so its norm is computed once and cached
    auto tc = Clock::now();
    const auto C = op_->solve(candidates);
    timings_["solve"] += elapsed(tc);
    std::vector<int64_t> fresh;
    for (int64_t d : candidates)
      if (std::isnan(rhs_norm_[d]))
        fresh.push_back(d);
    tc = Clock::now();
    const auto R = op_->residuals(candidates, C);
    torch::Tensor B;
    if (!fresh.empty())
      B = op_->residuals(
        fresh, torch::zeros({static_cast<int64_t>(fresh.size()), rank_}, F64));
    timings_["residual"] += elapsed(tc);
    tc = Clock::now();
    auto norm_of = [&](const std::vector<int64_t>& units,
                     const torch::Tensor& residuals) {
      return pairs() ? store_->pair_residual_norms(units, residuals)
                     : store_->residual_norms(units, residuals);
    };
    const auto norms = norm_of(candidates, R);
    if (!fresh.empty()) {
      const auto fresh_norms = norm_of(fresh, B);
      const double* fp = fresh_norms.data_ptr<double>();
      for (size_t i = 0; i < fresh.size(); i++)
        rhs_norm_[fresh[i]] = fp[i];
    }
    auto rhs = torch::empty({static_cast<int64_t>(candidates.size())}, F64);
    double* rp = rhs.data_ptr<double>();
    for (size_t i = 0; i < candidates.size(); i++)
      rp[i] = rhs_norm_[candidates[i]];
    timings_["norm"] += elapsed(tc);
    const double worst = norms.max().item<double>();
    const double scale = rhs.square().mean().sqrt().item<double>();
    const double ratio = scale > 0.0   ? worst / scale
                         : worst > 0.0 ? std::numeric_limits<double>::infinity()
                                       : 0.0;
    record.residual = ratio;
    error_ = ratio;

    // (3.7) passed: every direction's flux, and (3.9) on the scalar flux
    if (ratio <= config_.tol) {
      auto tf = Clock::now();
      const auto phi = fill();
      timings_["fill"] += elapsed(tf);
      filled = true;
      const bool compare =
        phi_old_.defined() && phi_old_.numel() == phi.numel();
      const double change = compare ? (phi - phi_old_).norm().item<double>() /
                                        std::max(phi.norm().item<double>(),
                                          std::numeric_limits<double>::min())
                                    : nan;
      record.phi_change = change;
      phi_old_ = phi;
      if (compare && change <= config_.tol) {
        converged_ = true;
        history_.push_back(record);
        break;
      }
    }
    history_.push_back(record);
    if (config_.max_iter > 0 && iter >= config_.max_iter)
      break;

    // Sweep the p candidates with the largest residuals
    const int64_t p = std::min<int64_t>(
      config_.per_iter, static_cast<int64_t>(candidates.size()));
    std::vector<int64_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    const double* n = norms.data_ptr<double>();
    std::partial_sort(order.begin(), order.begin() + p, order.end(),
      [&](int64_t a, int64_t b) { return n[a] > n[b]; });
    std::vector<int64_t> next;
    for (int64_t i = 0; i < p; i++)
      next.push_back(candidates[order[i]]);
    sample(next, "greedy");
  }
  // Stopped on max_iter after sampling more: fill on the final basis
  if (!filled) {
    auto tf = Clock::now();
    fill();
    timings_["fill"] += elapsed(tf);
  }
  if (config_.svd_tol > 0.0) {
    auto ts = Clock::now();
    truncate(false);
    timings_["svd"] += elapsed(ts);
  }
  if (config_.carry_basis) {
    auto tk = Clock::now();
    keep_basis();
    timings_["carry"] += elapsed(tk);
  }
  // The fibers are not needed once the flux is formed
  store_->release();
}

void GreedyGalerkin::compress()
{
  // DEIM-stage fluxes are not on an orthonormal basis: left as they are
  if (run_stage_ != RunStage::GREEDY || config_.svd_tol <= 0.0 ||
      !config_.keep_angular_flux || !coefficients_.defined() ||
      moments_.defined())
    return;
  auto ts = Clock::now();
  truncate(true);
  timings_["svd"] += elapsed(ts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t>
GreedyGalerkin::svd_cut(const torch::Tensor& Cw) const
{
  // The factors (linalg::truncated_svd()); the cut in svd_norm: FROBENIUS
  // is truncated_svd()'s own (relative), NUCLEAR the tail sum of the plain
  // singular values (truncated_svd() at delta = 0 only factors)
  const auto max_rank = std::numeric_limits<int64_t>::max();
  torch::Tensor P, s, Vh;
  int64_t k = 0;
  if (config_.svd_norm == SvdNorm::FROBENIUS) {
    std::tie(P, s, Vh, std::ignore, std::ignore) =
      linalg::truncated_svd(Cw, config_.svd_tol, max_rank, true);
    k = s.size(0);
  } else {
    std::tie(P, s, Vh, std::ignore, std::ignore) = linalg::truncated_svd(Cw);
    s = s.contiguous();
    const auto tail = s.flip(0).cumsum(0).flip(0); // tail[i] = sum_{j >= i}
    const double bound = config_.svd_tol * s.sum().item<double>();
    const double* t = tail.data_ptr<double>();
    k = s.size(0);
    while (k > 0 && t[k - 1] <= bound)
      k--;
  }
  return {P, s, Vh, k};
}

void GreedyGalerkin::truncate(bool replace)
{
  const int64_t r = coefficients_.size(1);
  if (r == 0) {
    truncated_rank_ = 0;
    return;
  }
  // The coefficients on the weighted orthonormal basis X~ with the angle
  // factor a_d: a_d C is then the flux in the norm of the stored fibers, and
  // its SVD the best truncation in that norm (Guo & Peng: C itself, every
  // direction weighted equally, which is weighting.angle = false)
  linalg::lapack::SingleThreaded single;
  const auto a = unit_weights().unsqueeze(1); // (num_units, 1)
  const auto [P, s, Vh, k] = svd_cut(coefficients_ * a);
  truncated_rank_ = k;
  if (!replace)
    return;

  // The paper's phi^(n,*): the moments of the untruncated fill
  moments_ = moments();
  coefficients_ =
    (P.narrow(1, 0, k) * s.narrow(0, 0, k).unsqueeze(0) / a).contiguous();
  basis_ = Vh.narrow(0, 0, k)
             .mm(basis_.reshape({r, -1}))
             .reshape({k, basis_groups_, num_space_})
             .contiguous();
}

void GreedyGalerkin::keep_basis()
{
  carry_coefficients_ = torch::Tensor();
  carry_rows_ = torch::Tensor();
  carry_unweighted_ = torch::Tensor();
  if (!coefficients_.defined() || coefficients_.size(1) == 0)
    return;
  // Copies: the run's buffers are overwritten by the next run (and the
  // reduced operator's basis dropped at release())
  const int64_t r = coefficients_.size(1);
  const auto a = unit_weights().unsqueeze(1);
  carry_coefficients_ = (coefficients_ * a).contiguous();
  carry_rows_ = basis_rows_.narrow(0, 0, r).clone();
  carry_unweighted_ = op_->basis().reshape({r, -1}).clone();
}

void GreedyGalerkin::start_from_kept_basis()
{
  if (!carry_coefficients_.defined())
    return;
  // a_d C on the weighted orthonormal basis X~: its right singular vectors
  // V rotate X~ into the directions that carry the last run's flux, ordered
  // by weight. Cut at this run's tolerance in the Frobenius norm, V^T X~
  // keeps that flux to a weighted L2 error of tol and stays orthonormal
  linalg::lapack::SingleThreaded single;
  const auto [U, s, Vh, delta, rest] =
    linalg::truncated_svd(carry_coefficients_, config_.tol,
      std::numeric_limits<int64_t>::max(), /*with_normalize=*/true);
  const int64_t k = Vh.size(0), L = basis_groups_ * num_space_;
  // The rotated rows become X~'s first rows (capacity doubles as in
  // orthonormalize()), and the reduced operator is bordered with all of
  // them at once (one BLAS-3 extension; it projects the new sources too)
  int64_t cap = basis_rows_.defined() ? basis_rows_.size(0) : 0;
  if (cap < k) {
    cap = std::max<int64_t>(4, cap);
    while (cap < k)
      cap *= 2;
    basis_rows_ = torch::zeros({cap, L}, F64);
  }
  basis_rows_.narrow(0, 0, k).copy_(Vh.mm(carry_rows_));
  auto t0 = Clock::now();
  op_->extend(Vh.mm(carry_unweighted_).view({k, basis_groups_, num_space_}));
  timings_["project"] += elapsed(t0);
  rank_ = k;
  carried_rank_ = k;
}

// =================================================================
// Han et al. (2026): the DEIM stage
// =================================================================

torch::Tensor GreedyGalerkin::unweighted_fibers(
  const std::vector<int64_t>& dirs) const
{
  // The stored fibers are a_d e_g R_P psi_{d,g}: undo each factor (the
  // store's own, 1 where disabled)
  const int64_t k = static_cast<int64_t>(dirs.size());
  const int64_t G = num_groups_, Ns = num_space_;
  auto out = torch::empty({k, G, Ns}, F64);
  double* o = out.data_ptr<double>();
  const auto& a = store_->angle_weights();
  const auto& e = store_->energy_weights();
  for (int64_t i = 0; i < k; i++)
    for (int64_t g = 0; g < G; g++) {
      const double* f = store_->fiber(dirs[i] * G + g);
      const double c = 1.0 / (a[dirs[i]] * e[g]);
      for (int64_t n = 0; n < Ns; n++)
        o[(i * G + g) * Ns + n] = c * f[n];
    }
  store_->apply_mass(o, k * G, true);
  return out;
}

torch::Tensor GreedyGalerkin::weighted_scalar_flux() const
{
  const int64_t G = num_groups_, Ns = num_space_;
  if (!coefficients_.defined() || coefficients_.size(1) == 0)
    return torch::zeros({G * Ns}, F64);
  const int64_t r = coefficients_.size(1);
  const auto wy = weights_ * harmonics_.select(1, 0);
  auto phi = coefficients_.t()
               .mv(wy)
               .unsqueeze(0)
               .mm(basis_.reshape({r, -1}))
               .reshape({G, Ns})
               .contiguous();
  store_->apply_mass(phi.data_ptr<double>(), G, false);
  const auto e = torch::tensor(store_->energy_weights(), F64).unsqueeze(1);
  return (phi * e).reshape({-1});
}

void GreedyGalerkin::select_directions(const torch::Tensor& V)
{
  // Eq. 3.5: a quadrature-weighted orthonormal basis of span(V), then the
  // rows; the interpolation coefficients V V(I, :)^{-1} map the swept
  // fibers to every direction (identity rows at I)
  linalg::lapack::SingleThreaded single;
  const auto Vw = linalg::weighted_orthonormal(V, weights_).first;
  deim_rows_ = config_.deim.select == DeimSelect::QDEIM
                 ? linalg::qdeim_select(Vw)
                 : linalg::deim_select(Vw);
  const auto VI = Vw.index_select(0, index_tensor(deim_rows_)); // (k, k)
  deim_interp_ = torch::linalg_solve(VI.t(), Vw.t()).t().contiguous();
  lebesgue_ = linalg::lebesgue_constant(Vw, deim_rows_, weights_);
}

void GreedyGalerkin::enter_deim()
{
  // Han et al., Alg. 3.1 step 17: the angular modes of the greedy flux, the
  // left singular vectors of its coefficients (with the angle factor a_d,
  // as truncate() weighs them), cut at eps_SVD (svd_tol in svd_norm, as
  // truncate() cuts; 0: every nonzero mode). The DEIM stage never changes
  // this rank (X = orth(Psi_I) has |I| vectors), so it bounds the stage's
  // accuracy: the run's own (relaxed) tolerance would freeze too few
  linalg::lapack::SingleThreaded single;
  const auto a = unit_weights().unsqueeze(1);
  auto [P, s, Vh, k] = svd_cut(coefficients_ * a);
  // Numerically zero modes carry nothing to interpolate
  const auto sv = s.contiguous();
  const double* sp = sv.data_ptr<double>();
  while (k > 1 && !(sp[k - 1] > DEIM_ZERO_MODE_TOL * sp[0]))
    k--;
  select_directions(P.narrow(1, 0, std::max<int64_t>(k, 1)) / a);
  // Han et al.'s outer test compares consecutive outer iterates: the first
  // update's predecessor is the greedy iterate of the switch
  update_flux_ = last_flux_;
  deim_ = true;
  inner_change_ = std::numeric_limits<double>::infinity();
}

void GreedyGalerkin::run_deim()
{
  reset();
  settled_ = false;
  run_stage_ = RunStage::DEIM_INNER;
  const int64_t G = num_groups_;
  std::vector<int64_t> fibers;
  for (int64_t d : deim_rows_)
    for (int64_t g = 0; g < G; g++)
      fibers.push_back(d * G + g);
  store_->ensure(fibers, "deim");

  // Inner source iteration: the swept fibers, every direction interpolated
  auto td = Clock::now();
  coefficients_ = deim_interp_;
  basis_ = unweighted_fibers(deim_rows_);
  rank_ = static_cast<int64_t>(deim_rows_.size()); // reported by ranks()
  auto phi = weighted_scalar_flux();
  // A diverged interpolated source iteration overflows, first in the norms
  // (Han et al.'s has DSA inside its inner iterations, this solver none)
  if (!std::isfinite(phi.norm().item<double>()))
    throw utils::runtime_error(ctx("run"),
      "The DEIM source iteration diverged (the flux norm overflows after " +
        std::to_string(updates_) +
        " angular updates); deim.update_on_growth and deim.galerkin_flux "
        "(ours) stabilize it");
  auto relative = [](const torch::Tensor& x, const torch::Tensor& ref) {
    return (x - ref).norm().item<double>() /
           std::max(
             x.norm().item<double>(), std::numeric_limits<double>::min());
  };
  run_change_ = last_flux_.defined() ? relative(phi, last_flux_)
                                     : std::numeric_limits<double>::infinity();
  error_ = run_change_;
  converged_ = true;
  const double inner =
    config_.deim.inner_tol > 0.0 ? config_.deim.inner_tol : config_.tol;
  // Recorded either way; with deim.update_on_growth (our safeguard, not Han
  // et al.'s) a growing change ends the inner loop early: interpolation
  // amplifies the error outside span(V) by up to the Lebesgue constant, so
  // with strong scattering the interpolated source iteration can diverge
  inner_diverged_ = run_change_ > inner_change_;
  timings_["deim"] += elapsed(td);

  // Converged inner iteration: the angular update. X = orth(Psi_I) (MGS2-A
  // of the swept snapshots: the fibers are stored, nothing is swept again;
  // the reduced operator is bordered with X and projects this solve's
  // sources and scattering), every direction solved with its scattering,
  // then V and I from that solution. Han et al. keep the interpolant as the
  // iterate (phi^(n+1) = phi^(n+1,l)); deim.galerkin_flux (ours) holds X L
  if (run_change_ <= inner ||
      (config_.deim.update_on_growth && inner_diverged_)) {
    run_stage_ = RunStage::DEIM_UPDATE;
    inner_change_ = std::numeric_limits<double>::infinity();
    auto held_coefficients = coefficients_;
    auto held_basis = basis_;
    const int64_t held_rank = rank_;
    rank_ = 0; // X~ starts empty
    sample(deim_rows_, "deim");
    if (rank_ == 0)
      throw utils::runtime_error(ctx("run"),
        "The DEIM directions' fibers are zero or overflow (a diverged "
        "source iteration?): no basis for the angular update");
    auto tu = Clock::now();
    auto [L, Lam] = op_->angular_galerkin(op_->fixed_source());
    if (config_.deim.galerkin_flux) {
      coefficients_ = L;
      basis_ = op_->basis();
      phi = weighted_scalar_flux();
    } else {
      coefficients_ = std::move(held_coefficients);
      basis_ = std::move(held_basis);
      rank_ = held_rank;
    }
    outer_change_ = update_flux_.defined()
                      ? relative(phi, update_flux_)
                      : std::numeric_limits<double>::quiet_NaN();
    update_flux_ = phi;
    // Han et al.'s outer test, ||phi^(n) - phi^(n-1)|| <= eps_phi: the
    // solver's own stop tolerance (the run's tolerance is a fraction of it)
    const double stop = stop_tol_ > 0.0 ? stop_tol_ : config_.tol;
    settled_ = !std::isnan(outer_change_) && outer_change_ <= stop;
    updates_++;
    select_directions(L);
    timings_["update"] += elapsed(tu);
  } else {
    inner_change_ = run_change_;
  }
  last_flux_ = phi;
  store_->release();
}

torch::Tensor GreedyGalerkin::moments() const
{
  if (!holds_flux())
    throw utils::runtime_error(ctx("moments"), "No flux held (run first)");
  if (moments_.defined())
    return moments_;
  const int64_t r = coefficients_.size(1);
  const int64_t M = harmonics_.size(1);
  if (r == 0)
    return torch::zeros({M, num_groups_, num_space_}, F64);
  // phi_m = X (sum_d w_d Y_m(d) c_d); for pairs per group g, phi_{m,g} =
  // X (sum_d w_d Y_m(d) c_{d,g}) with X spatial
  const auto HW = harmonics_.t() * weights_.unsqueeze(0); // (M, N_Omega)
  if (!pairs())
    return HW.mm(coefficients_)
      .mm(basis_.reshape({r, -1}))
      .view({M, num_groups_, num_space_});
  const auto B = torch::einsum(
    "md,dgr->mgr", {HW, coefficients_.view({num_directions_, num_groups_, r})});
  return B.reshape({M * num_groups_, r})
    .mm(basis_.reshape({r, -1}))
    .view({M, num_groups_, num_space_});
}

torch::Tensor GreedyGalerkin::evaluate(const std::vector<int64_t>& directions,
  const std::vector<int64_t>& groups, const std::vector<int64_t>& space) const
{
  if (!holds_flux())
    throw utils::runtime_error(ctx("evaluate"), "No flux held (run first)");
  const int64_t D = static_cast<int64_t>(directions.size());
  const int64_t ng = static_cast<int64_t>(groups.size());
  const int64_t ns = static_cast<int64_t>(space.size());
  if (coefficients_.size(1) == 0)
    return torch::zeros({D, ng, ns}, F64);
  if (pairs()) {
    // Unit d G + g per (direction, group), on the spatial basis
    std::vector<int64_t> units;
    for (int64_t d : directions)
      for (int64_t g : groups)
        units.push_back(d * num_groups_ + g);
    const auto C =
      coefficients_.index_select(0, index_tensor(units)).view({D, ng, -1});
    const auto X = basis_.select(1, 0).index_select(1, index_tensor(space));
    return torch::einsum("dgr,rs->dgs", {C, X});
  }
  const auto C = coefficients_.index_select(0, index_tensor(directions));
  const auto X = basis_.index_select(1, index_tensor(groups))
                   .index_select(2, index_tensor(space));
  return torch::einsum("dr,rgs->dgs", {C, X});
}

void GreedyGalerkin::release()
{
  coefficients_ = torch::Tensor();
  basis_ = torch::Tensor();
  moments_ = torch::Tensor();
  op_->reset();
}

std::map<std::string, double> GreedyGalerkin::timings() const
{
  auto out = timings_;
  for (const auto& [k, v] : store_->timings())
    out[k] += v;
  return out;
}

sampling::RunRecord::Ptr GreedyGalerkin::record() const
{
  auto out = std::make_shared<RunRecord>();
  fill_record(*out);
  out->history = history_;
  out->dropped_snapshots = run_dropped_;
  out->orthogonality_failures = run_failures_;
  out->truncated_rank = truncated_rank_;
  out->carried_rank = carried_rank_;
  out->stage = run_stage_;
  out->run_change = run_change_;
  out->deim_directions = run_stage_ == RunStage::GREEDY
                           ? 0
                           : static_cast<int64_t>(store_->num_swept()) /
                               std::max<int64_t>(1, num_groups_);
  out->lebesgue = lebesgue_;
  out->outer_change = outer_change_;
  out->stop_tolerance = stop_tol_ > 0.0 ? stop_tol_ : config_.tol;
  out->settled = settled_;
  out->inner_diverged = run_stage_ != RunStage::GREEDY && inner_diverged_;
  return out;
}

} // namespace ttnte::galerkin
