#include "ttnte/galerkin/greedy_galerkin.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace ttnte::galerkin {

namespace {

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

GreedyGalerkin::GreedyGalerkin(
  cross::CrossProblem problem, ReducedOperator::Ptr op, GalerkinConfig config)
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
  if (config_.max_iter < 0)
    throw utils::runtime_error(ctx("GreedyGalerkin"), "max_iter must be >= 0");
  if (config_.phi_old == GalerkinPhiOld::AUTO)
    throw utils::runtime_error(ctx("GreedyGalerkin"),
      "phi_old must be resolved (PREVIOUS_RUN or WITHIN_RUN) before "
      "construction");
  weights_ = problem.weights.to(torch::kFloat64).contiguous();
  harmonics_ = problem.harmonics.to(torch::kFloat64).contiguous();
  initial_directions_ = problem.initial_directions;
  store_ = cross::FiberStore::create(std::move(problem), config_.weighting);
  num_directions_ = op_->num_directions();
  num_groups_ = store_->problem().num_groups;
  num_space_ = store_->problem().num_space;
  if (weights_.size(0) != num_directions_ ||
      harmonics_.size(0) != num_directions_)
    throw utils::runtime_error(ctx("GreedyGalerkin"),
      "The problem's and the operator's directions differ");
  swept_.assign(static_cast<size_t>(num_directions_), 0);
  coordinates_.resize(static_cast<size_t>(num_directions_));
}

std::vector<int64_t> GreedyGalerkin::draw(int64_t n)
{
  std::vector<int64_t> pool;
  for (int64_t d = 0; d < num_directions_; d++)
    if (!swept_[d])
      pool.push_back(d);
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

std::vector<double> GreedyGalerkin::orthonormalize(torch::Tensor psi)
{
  // MGS2-A (Alg. 3.2) of one snapshot against the rank_ rows of X~:
  // two projections, then up to mgs_max_projections - 2 cleanups while
  // |X(:, 1)^T psi| > mgs_eps. Invariant: snapshot = X c + alpha psi.
  const int64_t L = num_groups_ * num_space_;
  const double norm0 = psi.norm().item<double>();
  auto c = torch::zeros({rank_}, F64);
  const auto X = basis_rows_.defined() ? basis_rows_.narrow(0, 0, rank_)
                                       : torch::empty({0, L}, F64);
  auto project = [&] {
    const auto dc = X.mv(psi);
    psi.addmv_(X.t(), dc, 1.0, -1.0);
    c += dc;
  };
  if (rank_ > 0) {
    project();
    project();
  }
  double alpha = psi.norm().item<double>();
  std::vector<double> out(static_cast<size_t>(rank_));
  // Not appended (an addition to Alg. 3.2, which always appends): a
  // snapshot already in the span to basis_rtol of its norm, a zero one, or
  // any once the basis spans the whole space. Its remainder would be
  // rounding noise, normalized into a spurious vector (and past the
  // space's dimension, a dependent one: a singular reduced matrix). Its
  // coordinates c reproduce it to that tolerance.
  auto in_span = [&] {
    return alpha <= config_.basis_rtol * norm0 || rank_ >= L;
  };
  if (alpha == 0.0 || in_span()) {
    if (norm0 > 0.0)
      dropped_++;
    std::copy_n(c.data_ptr<double>(), rank_, out.begin());
    return out;
  }
  psi /= alpha;
  if (rank_ > 0) {
    auto first = [&] { return std::abs(X[0].dot(psi).item<double>()); };
    for (int64_t m = 0; m < config_.mgs_max_projections - 2; m++) {
      if (first() <= config_.mgs_eps)
        break;
      psi *= alpha;
      project();
      alpha = psi.norm().item<double>();
      if (alpha == 0.0 || in_span()) {
        dropped_++;
        std::copy_n(c.data_ptr<double>(), rank_, out.begin());
        return out;
      }
      psi /= alpha;
    }
    if (first() > config_.mgs_eps)
      orthogonality_failures_++;
  }
  // Append psi as row rank_ (capacity doubles)
  const int64_t cap = basis_rows_.defined() ? basis_rows_.size(0) : 0;
  if (rank_ == cap) {
    auto grown = torch::zeros({std::max<int64_t>(4, 2 * cap), L}, F64);
    if (rank_ > 0)
      grown.narrow(0, 0, rank_).copy_(basis_rows_.narrow(0, 0, rank_));
    basis_rows_ = grown;
  }
  basis_rows_[rank_].copy_(psi);
  std::copy_n(c.data_ptr<double>(), rank_, out.begin());
  out.push_back(alpha);
  rank_++;
  return out;
}

void GreedyGalerkin::sample(
  const std::vector<int64_t>& directions, const char* phase)
{
  const int64_t G = num_groups_, Ns = num_space_;
  std::vector<int64_t> pairs;
  for (int64_t d : directions)
    for (int64_t g = 0; g < G; g++)
      pairs.push_back(d * G + g);
  store_->ensure(pairs, phase);

  // Snapshots: the stored fibers a_d e_g R_P psi_{d,g} without the angle
  // factor a_d. The span would not care, but a swept direction's MGS
  // coordinates become its row of C (Remark 3.3), so they must reproduce
  // e_g R_P psi_{d,g} itself; the angle factor enters only the residual
  // norms. a_d is the store's own factor (1 with angle weighting off), so
  // this undoes exactly what the store applied. Orthonormalized in turn
  // (each new vector orthogonalizes the next, as in Alg. 3.2)
  auto t0 = Clock::now();
  const int64_t r0 = rank_;
  const auto& angle = store_->angle_weights();
  for (int64_t d : directions) {
    auto psi = torch::empty({G * Ns}, F64);
    double* p = psi.data_ptr<double>();
    for (int64_t g = 0; g < G; g++)
      std::copy_n(store_->fiber(d * G + g), Ns, p + g * Ns);
    psi /= angle[d];
    coordinates_[d] = orthonormalize(psi);
    swept_[d] = 1;
  }
  timings_["orth"] += elapsed(t0);

  // The new rows, unweighted (R_P^{-1}, 1 / e_g) for the reduced operator
  t0 = Clock::now();
  const int64_t p = rank_ - r0;
  if (p > 0) {
    auto added = basis_rows_.narrow(0, r0, p).clone().view({p, G, Ns});
    store_->apply_mass(added.data_ptr<double>(), p * G, true);
    const auto& energy = store_->energy_weights();
    for (int64_t g = 0; g < G; g++)
      added.select(1, g).div_(energy[g]);
    op_->extend(added);
  }
  timings_["project"] += elapsed(t0);
}

torch::Tensor GreedyGalerkin::fill()
{
  const int64_t D = num_directions_, r = rank_;
  auto C = torch::zeros({D, r}, F64);
  if (r > 0) {
    std::vector<int64_t> unswept;
    for (int64_t d = 0; d < D; d++)
      if (!swept_[d])
        unswept.push_back(d);
    if (!unswept.empty())
      C.index_copy_(0, index_tensor(unswept), op_->solve(unswept));
    // Swept directions: their exact fibers (Remark 3.3), zero beyond the
    // rank at which they were added (later vectors are orthogonal to them)
    double* c = C.data_ptr<double>();
    for (int64_t d = 0; d < D; d++)
      if (swept_[d])
        std::copy(coordinates_[d].begin(), coordinates_[d].end(), c + d * r);
  }
  coefficients_ = C;
  basis_ = op_->basis();
  // Weighted scalar flux X~ a, a = C^T (w Y_0): the space and energy factors
  // only (X~ carries no angle factor)
  const auto a = C.t().mv(weights_ * harmonics_.select(1, 0));
  if (r == 0)
    return torch::zeros({num_groups_ * num_space_}, F64);
  return basis_rows_.narrow(0, 0, r).t().mv(a);
}

void GreedyGalerkin::run()
{
  const auto t0 = Clock::now();
  store_->clear();
  op_->reset();
  rank_ = 0;
  std::fill(swept_.begin(), swept_.end(), 0);
  for (auto& c : coordinates_)
    c.clear();
  coefficients_ = torch::Tensor();
  basis_ = torch::Tensor();
  history_.clear();
  // (3.9)'s reference: the previous run's flux (carried) or none yet
  if (config_.phi_old == GalerkinPhiOld::WITHIN_RUN)
    phi_old_ = torch::Tensor();
  converged_ = false;
  error_ = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Start: p random directions, or the cross's initial ones
  std::vector<int64_t> start;
  if (config_.start == GalerkinStart::INITIAL_DIRECTIONS) {
    for (int64_t d : initial_directions_)
      if (d >= 0 && d < num_directions_ &&
          std::find(start.begin(), start.end(), d) == start.end())
        start.push_back(d);
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
    // right-hand sides (residuals of zero) for the pooled scale
    auto tc = Clock::now();
    const auto C = op_->solve(candidates);
    timings_["solve"] += elapsed(tc);
    tc = Clock::now();
    const auto R = op_->residuals(candidates, C);
    const auto B = op_->residuals(candidates,
      torch::zeros({static_cast<int64_t>(candidates.size()), rank_}, F64));
    timings_["residual"] += elapsed(tc);
    tc = Clock::now();
    const auto norms = store_->residual_norms(candidates, R);
    const auto rhs = store_->residual_norms(candidates, B);
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
  // The fibers are not needed once the flux is formed
  store_->release();
  seconds_ = elapsed(t0);
}

torch::Tensor GreedyGalerkin::moments() const
{
  if (!holds_flux())
    throw utils::runtime_error(ctx("moments"), "No flux held (run first)");
  const int64_t r = coefficients_.size(1);
  const int64_t M = harmonics_.size(1);
  if (r == 0)
    return torch::zeros({M, num_groups_, num_space_}, F64);
  // phi_m = X (sum_d w_d Y_m(d) c_d)
  const auto B = (harmonics_.t() * weights_.unsqueeze(0)).mm(coefficients_);
  return B.mm(basis_.reshape({r, -1})).view({M, num_groups_, num_space_});
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
  const auto C = coefficients_.index_select(0, index_tensor(directions));
  const auto X = basis_.index_select(1, index_tensor(groups))
                   .index_select(2, index_tensor(space));
  return torch::einsum("dr,rgs->dgs", {C, X});
}

void GreedyGalerkin::release()
{
  coefficients_ = torch::Tensor();
  basis_ = torch::Tensor();
  op_->reset();
}

std::map<std::string, double> GreedyGalerkin::timings() const
{
  auto out = timings_;
  for (const auto& [k, v] : store_->timings())
    out[k] += v;
  return out;
}

} // namespace ttnte::galerkin
