// SweepSolver: Outer iterations, certification, solutions and diagnostics.
#include "ttnte/sweep/sweep_driver.hpp"

namespace ttnte::sweep {

namespace {

/// Relative accuracy of the TT a Solution stores each patch's flux in.
constexpr double SOLUTION_TT_EPS = 1e-14;
/// Largest spectral-radius estimate used by rho_safety (the target stays
/// positive).
constexpr double RHO_CAP = 0.999;

} // namespace

typename SweepSolver::Solution::Ptr SweepSolver::make_solution(
  bool angular) const
{
  auto solution = Solution::create(
    serial_ ? parallel::Communicator::self() : parallel::Communicator::world(),
    mesh_, qset_, xs_server_, physics::DGTransportAssemblerConfig(),
    eigenvalue_mode_ ? std::optional<double>(k_) : std::nullopt, angular);
  solution->set_iteration_counts(
    static_cast<int>(num_iterations_), static_cast<int>(num_iterations_));
  for (const auto& P : patches_) {
    auto dense = angular ? get_angular_flux(P.gid) : get_scalar_flux(P.gid);
    solution->add_local_patch(P.gid,
      linalg::State(linalg::TTEngine::from_dense(dense, SOLUTION_TT_EPS)));
  }
  return solution;
}

std::vector<double> SweepSolver::scalar_flux_snapshot() const
{
  std::vector<double> phi;
  for (const auto& P : patches_) {
    const double* m = P.moments.template data_ptr<double>();
    phi.insert(phi.end(), m, m + num_groups_ * P.disc->num_dofs());
  }
  return phi;
}

double SweepSolver::scalar_flux_change(
  const std::vector<double>& old, bool relative_to_old) const
{
  const auto phi = scalar_flux_snapshot();
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < phi.size(); i++) {
    num += (phi[i] - old[i]) * (phi[i] - old[i]);
    den += relative_to_old ? old[i] * old[i] : phi[i] * phi[i];
  }
  num = global_sum(num);
  den = global_sum(den);
  return den > 0 ? std::sqrt(num / den) : 0.0;
}

double SweepSolver::fission_production() const
{
  double total = 0.0;
  for (const auto& P : patches_) {
    if (!P.fission.defined())
      continue;
    total +=
      transfer_total(P.disc->view(), P.fission.template data_ptr<double>(),
        num_groups_, P.moments.template data_ptr<double>(),
        P.disc->basis_integral().template data_ptr<double>());
  }
  return global_sum(total);
}

void SweepSolver::update_fission_source()
{
  thread_local std::vector<double> scratch;
  std::vector<int64_t> all(num_groups_);
  std::iota(all.begin(), all.end(), 0);
  for (auto& P : patches_) {
    if (!P.fission.defined())
      continue;
    P.fission_source.zero_();
    transfer_source(P.disc->view(), P.fission.template data_ptr<double>(), 1,
      moment_order_raw_.data(), 1, num_groups_,
      P.moments.template data_ptr<double>(), 1.0 / k_, all.data(), num_groups_,
      nullptr, P.fission_source.template data_ptr<double>(), scratch);
  }
}

void SweepSolver::certify()
{
  const auto t0 = std::chrono::steady_clock::now();
  const auto old = scalar_flux_snapshot();
  if (eigenvalue_mode_)
    update_fission_source();
  for (size_t s = 0; s < group_indices_.size(); s++) {
    compute_source(s, Within::MOMENTS, true, false);
    sweep_group_set(s);
  }
  for (size_t s = 0; s < group_indices_.size(); s++)
    commit_group_set(s);
  certification_error_ = scalar_flux_change(old, false);
  if (eigenvalue_mode_) {
    const double k_new = fission_production();
    certification_k_change_ = std::abs(k_new - k_);
    k_ = k_new;
  }
  certification_seconds_ =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

void SweepSolver::outer_iterations(double k_tol)
{
  // Sweep tasks are the parallelism: torch inside them runs serially
  ThreadCountGuard threads(config_.num_threads > 1 ? 1 : 0);

  const bool verbose = config_.verbose && comm_.rank() == 0;
  auto start = std::chrono::steady_clock::now();
  solve_start_ = std::chrono::steady_clock::now();
  last_progress_ = solve_start_;
  double callback_seconds = 0.0;
  solve_seconds_ = 0.0;
  double previous_error = -1.0;
  if (eigenvalue_mode_) {
    // Normalize the initial flux to unit fission production with k = 1, so
    // the first iterate (fission source / k) has the scale of the warm
    // start; an initial k far from the eigenvalue makes a loosely solved
    // first inner (inner_forcing) meaningless.
    const double production = fission_production();
    for (auto& P : patches_)
      P.moments.div_(production);
    k_ = 1.0;
  }

  int64_t it = 0;
  for (; it < config_.max_iter; it++) {
    const auto old = scalar_flux_snapshot();
    if (eigenvalue_mode_)
      update_fission_source();

    const double inner_tol = inner_tolerance();
    auto t0 = std::chrono::steady_clock::now();
    outer_sweeps_.push_back(group_set_pass());
    sweep_times_.push_back(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count());

    double error, k_error = 0.0;
    if (eigenvalue_mode_) {
      // As TransportDriver::solve_eigenvalue(): relative to the old flux
      error = scalar_flux_change(old, true);
      const double k_new = fission_production();
      k_error = std::abs(k_new - k_);
      k_ = k_new;
      k_history_.push_back(k_);
    } else {
      error = scalar_flux_change(old, false);
    }
    errors_.push_back(error);
    const double rho = previous_error > 0 ? error / previous_error : 0.0;
    previous_error = error;
    double target = config_.tol;
    if (config_.rho_safety)
      target *= (1.0 - std::min(rho, RHO_CAP));

    if (verbose) {
      std::cout << "-- outer (" << it << "): ";
      if (eigenvalue_mode_)
        std::cout << "k = " << std::fixed << std::setprecision(6) << k_
                  << ", dk = " << std::scientific << std::setprecision(2)
                  << k_error << ", ";
      std::cout << "scalar flux error = " << std::scientific
                << std::setprecision(4) << error << ", rho = " << std::fixed
                << std::setprecision(4) << rho
                << ", group-set sweeps = " << outer_sweeps_.back()
                << ", inner tol = " << std::scientific << std::setprecision(1)
                << inner_tol << std::fixed
                << ", elapsed = " << std::setprecision(3)
                << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
                << " s" << std::defaultfloat << std::endl;
    }
    solve_seconds_ =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
        .count() -
      callback_seconds;
    if (callback_ && it % callback_frequency_ == 0) {
      auto c0 = std::chrono::steady_clock::now();
      callback_(*this);
      callback_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - c0)
          .count();
    }
    const bool k_ok = !eigenvalue_mode_ || k_error < k_tol;
    if (it > 0 && error < target && k_ok) {
      converged_ = true;
      it++;
      break;
    }
  }
  num_iterations_ = std::min<int64_t>(it, config_.max_iter);
  if (config_.certify && config_.transport == TransportSolve::CROSS) {
    certify();
    if (verbose) {
      std::cout << "-- certification: exact transport solve, scalar flux "
                   "change = "
                << std::scientific << std::setprecision(3)
                << certification_error_;
      if (eigenvalue_mode_)
        std::cout << ", k change = " << certification_k_change_;
      std::cout << std::defaultfloat << " (" << certification_seconds_ << " s)"
                << std::endl;
    }
  }
  if (!config_.cross.keep_angular_flux)
    for (auto& cr : crosses_)
      cr->release();
  if (verbose) {
    std::cout << "-- " << (converged_ ? "Converged!" : "Failed to Converge!")
              << " (" << num_iterations_ << " outer iterations, "
              << equivalent_sweeps_ << " sweep equivalents)" << std::endl;
  }
}

cross::LayoutCross::Ptr SweepSolver::get_cross(size_t s) const
{
  if (s >= crosses_.size())
    throw utils::runtime_error(
      ctx("get_cross"), "No cross for this group set (transport = CROSS?)");
  return crosses_[s];
}

std::map<std::string, double> SweepSolver::get_cross_timings() const
{
  auto out = cross_timings_;
  for (const auto& cr : crosses_)
    for (const auto& [phase, sec] : cr->timings())
      out[phase] += sec;
  return out;
}

std::vector<int64_t> SweepSolver::local_gids() const
{
  std::vector<int64_t> gids;
  gids.reserve(patches_.size());
  for (const auto& P : patches_)
    gids.push_back(P.gid);
  return gids;
}

typename SweepSolver::Solution::Ptr SweepSolver::solve_fixed_source()
{
  if (!is_setup_)
    throw utils::runtime_error(ctx("solve_fixed_source"), "Call setup() first");
  if (has_fission_)
    throw utils::runtime_error(ctx("solve_fixed_source"),
      "Fissile materials need solve_eigenvalue(); subcritical "
      "multiplication (fixed source + fission) is not supported");
  eigenvalue_mode_ = false;
  reset_state(false);
  outer_iterations(0.0);
  return config_.build_solution ? build_solution() : nullptr;
}

typename SweepSolver::Solution::Ptr SweepSolver::solve_eigenvalue(double k_tol)
{
  if (!is_setup_)
    throw utils::runtime_error(ctx("solve_eigenvalue"), "Call setup() first");
  if (!has_fission_)
    throw utils::runtime_error(
      ctx("solve_eigenvalue"), "No fissile material in the mesh");
  // Decided on the global mesh at setup, so every rank throws together
  if (has_external_)
    throw utils::runtime_error(ctx("solve_eigenvalue"),
      "Fixed sources are not supported in eigenvalue problems");
  eigenvalue_mode_ = true;
  reset_state(true);
  outer_iterations(k_tol);
  return config_.build_solution ? build_solution() : nullptr;
}

torch::Tensor SweepSolver::get_scalar_flux(int64_t gid) const
{
  const auto& P = patches_.at(gid_to_local_.at(gid));
  auto shape = P.disc->shape();
  shape.push_back(num_groups_);
  return P.moments[0].transpose(0, 1).reshape(shape).contiguous();
}

torch::Tensor SweepSolver::get_angular_flux(int64_t gid) const
{
  const size_t local = gid_to_local_.at(gid);
  const auto& P = patches_.at(local);
  torch::Tensor psi = P.psi;
  if (!psi.defined()) {
    // A cross that kept its TT evaluates the patch's flux from it
    const bool held =
      !crosses_.empty() && std::all_of(crosses_.begin(), crosses_.end(),
                             [](const auto& c) { return c->has_tt(); });
    if (!held) {
      throw utils::runtime_error(ctx("get_angular_flux"),
        "The angular flux is only kept with store_angular_flux = true, or "
        "from the TT with transport = CROSS and cross.keep_angular_flux = "
        "true");
    }
    psi = torch::empty(
      {omega_.size(0), num_groups_, P.disc->num_dofs()}, torch::kFloat64);
    for (size_t s = 0; s < crosses_.size(); s++)
      psi.index_copy_(1, torch::tensor(group_indices_[s], torch::kInt64),
        cross_patch_psi(s, local));
  }
  std::vector<int64_t> shape = angular_modes();
  for (auto s : P.disc->shape())
    shape.push_back(s);
  shape.push_back(num_groups_);
  return psi.permute({0, 2, 1}).reshape(shape).contiguous();
}

torch::Tensor SweepSolver::assemble_dense_operator(int64_t gid) const
{
  const auto& P = patches_.at(gid_to_local_.at(gid));
  const auto& disc = *P.disc;
  const int64_t D = omega_.size(0);
  const int64_t N = disc.num_dofs();
  const int64_t G = num_groups_;
  auto A = torch::zeros({D, N, G, D, N, G}, torch::kFloat64);

  // Streaming, collision and outflow (block diagonal in angle and group)
  auto values = disc.streaming_values(omega_);
  for (int64_t g = 0; g < G; g++) {
    auto L = disc.to_dense(
      values + P.sigma_t[g].template item<double>() * disc.mass_values());
    for (int64_t d = 0; d < D; d++) {
      A.index_put_(
        {d, torch::indexing::Slice(), g, d, torch::indexing::Slice(), g}, L[d]);
    }
  }

  // Scattering: K(d, d') w_d' sigma_l(g, g') M
  auto M = disc.to_dense(disc.mass_values().unsqueeze(0))[0];
  const int64_t L = P.scatter.defined() ? P.scatter.size(0) : 0;
  for (int64_t m = 0; m < harmonics_.size(1); m++) {
    int64_t l = moment_order_[m].template item<int64_t>();
    if (l >= L)
      continue;
    auto K = torch::outer(
      harmonics_.select(1, m), weighted_harmonics_.select(1, m)); // (D, D')
    A -= torch::einsum("ab,vu,gh->avgbuh", {K, M, P.scatter[l]});
  }

  // Reflective inflow: rows d, columns reflected d*
  for (size_t f = 0; f < P.links.size(); f++) {
    const auto& link = P.links[f];
    if (link.type != physics::BoundaryType::REFLECTIVE)
      continue;
    const auto& face = disc.faces()[f];
    auto c = link.albedo *
             face.upwind_weights(omega_.index_select(0, link.reflected), true);
    for (int64_t d = 0; d < D; d++) {
      int64_t ds = link.reflected[d].template item<int64_t>();
      auto B = torch::zeros({N, N}, torch::kFloat64);
      auto vals = c[d].index({face.pair_q}) * face.pair_val;
      B.index_put_(
        {face.dofs.index({face.pair_a}), face.dofs.index({face.pair_b})}, vals,
        /*accumulate=*/true);
      for (int64_t g = 0; g < G; g++) {
        A.index(
           {d, torch::indexing::Slice(), g, ds, torch::indexing::Slice(), g})
          .sub_(B);
      }
    }
  }
  return A.reshape({D * N * G, D * N * G});
}

void SweepSolver::set_callback(Callback callback, int64_t frequency)
{
  if (frequency < 1) {
    throw utils::runtime_error(
      ctx("set_callback"), "`frequency` must be at least 1");
  }
  callback_ = std::move(callback);
  callback_frequency_ = frequency;
}

std::map<std::string, int64_t> SweepSolver::memory_report() const
{
  std::map<std::string, int64_t> bytes;
  auto add = [&](const std::string& key, const torch::Tensor& t) {
    if (t.defined())
      bytes[key] += t.numel() * t.element_size();
  };
  for (const auto& P : patches_) {
    add("angular_flux", P.psi);
    for (const auto& t : P.trace_current)
      add("traces", t);
    for (const auto& t : P.trace_previous)
      add("traces", t);
    for (const auto& t : P.ghost_current)
      add("ghost_traces", t);
    for (const auto& t : P.ghost_previous)
      add("ghost_traces", t);
    for (const auto* t : {&P.moments, &P.moments_next, &P.source_moments,
           &P.krylov_moments, &P.pass_moments, &P.fission_source})
      add("moments", *t);
    add("factorizations", P.factors);
    add("factorizations", P.factor_pivots);
  }
  int64_t krylov = 0;
  for (size_t s = 0; s < group_indices_.size(); s++)
    krylov = std::max(krylov, krylov_size(s));
  bytes["krylov_vector"] = krylov * 8;
  bytes["gmres"] = config_.inner_solver == InnerSolver::GMRES
                     ? (config_.gmres_restart + 4) * krylov * 8
                     : 0;
  return bytes;
}

std::tuple<int64_t, int64_t, int64_t> SweepSolver::iterative_statistics() const
{
  int64_t solves = 0, applies = 0, failures = 0;
  for (const auto& P : patches_)
    if (P.iterative) {
      solves += P.iterative->num_solves();
      applies += P.iterative->num_iterations();
      failures += P.iterative->num_failures();
    }
  return {solves, applies, failures};
}

} // namespace ttnte::sweep
