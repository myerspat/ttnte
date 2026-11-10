// SweepSolver: Within-group-set solvers: Richardson, GMRES, correction form,
// incremental transport.
#include "ttnte/sweep/sweep_driver.hpp"

namespace ttnte::sweep {

double SweepSolver::global_sum(double value) const
{
  if (comm_.size() == 1)
    return value;
  double sum = 0.0;
  comm_.iallreduce(&value, &sum, 1, parallel::MPIOp::SUM).wait();
  return sum;
}

void SweepSolver::compute_source(
  size_t s, Within within, bool fixed, bool jacobi)
{
  thread_local std::vector<double> scratch;
  const auto& groups = group_indices_[s];
  const int64_t ng = static_cast<int64_t>(groups.size());
  const int64_t M = harmonics_.size(1);
  const int64_t G = num_groups_;
  for_each_patch([&](PatchState& P) {
    const auto& view = P.disc->view();
    const int64_t N = view.num_dofs;
    double* out = P.source_moments.template data_ptr<double>();
    for (int64_t m = 0; m < M; m++)
      for (int64_t g : groups)
        std::fill(out + (m * G + g) * N, out + (m * G + g + 1) * N, 0.0);

    const bool scatters = P.scatter.defined();
    const double* sigma_s =
      scatters ? P.scatter.template data_ptr<double>() : nullptr;
    const int64_t L = scatters ? P.scatter.size(0) : 0;
    if (fixed) {
      const double* q = nullptr;
      if (eigenvalue_mode_ && P.fission_source.defined())
        q = P.fission_source.template data_ptr<double>();
      else if (!eigenvalue_mode_ && P.external.defined())
        q = P.external.template data_ptr<double>();
      if (q) {
        for (int64_t g : groups)
          for (int64_t i = 0; i < N; i++)
            out[g * N + i] += q[g * N + i];
      }
      if (scatters && static_cast<int64_t>(group_indices_.size()) > 1) {
        const auto& other = jacobi ? P.pass_moments : P.moments;
        transfer_source(view, sigma_s, L, moment_order_raw_.data(), M, G,
          other.template data_ptr<double>(), 1.0, groups.data(), ng,
          group_complements_[s].data(), out, scratch);
      }
    }
    if (within != Within::NONE && scatters) {
      const auto& phi = within == Within::KRYLOV ? P.krylov_moments : P.moments;
      transfer_source(view, sigma_s, L, moment_order_raw_.data(), M, G,
        phi.template data_ptr<double>(), 1.0, groups.data(), ng,
        group_masks_[s].data(), out, scratch);
    }
  });
}

void SweepSolver::commit_group_set(size_t s)
{
  const auto& groups = group_indices_[s];
  const int64_t M = harmonics_.size(1);
  const int64_t G = num_groups_;
  for_each_patch([&](PatchState& P) {
    const int64_t N = P.disc->num_dofs();
    const double* next = P.moments_next.template data_ptr<double>();
    double* cur = P.moments.template data_ptr<double>();
    for (int64_t m = 0; m < M; m++)
      for (int64_t g : groups)
        std::copy(next + (m * G + g) * N, next + (m * G + g + 1) * N,
          cur + (m * G + g) * N);
  });
  // Previous traces are read only where they are lagged (reflective faces
  // and LAGGED interface edges, the delayed blocks), so only those
  // directions are carried over
  parallel_for(static_cast<int64_t>(delayed_.size()), [&](int64_t b) {
    const auto& blk = delayed_[b];
    const auto& current = delayed_buffer(blk, true);
    const int64_t Nf = current.size(2);
    const double* src = current.template data_ptr<double>();
    double* dst = delayed_buffer(blk, false).template data_ptr<double>();
    for (int64_t d : blk.directions)
      for (int64_t g : groups)
        std::copy(src + (d * G + g) * Nf, src + (d * G + g + 1) * Nf,
          dst + (d * G + g) * Nf);
  });
}

int64_t SweepSolver::krylov_size(size_t s) const
{
  const int64_t ng = static_cast<int64_t>(group_indices_[s].size());
  const int64_t M = harmonics_.size(1);
  int64_t n = 0;
  for (const auto& P : patches_)
    n += M * ng * P.disc->num_dofs();
  for (const auto& blk : delayed_)
    n += static_cast<int64_t>(blk.directions.size()) * ng *
         delayed_buffer(blk, false).size(2);
  return n;
}

void SweepSolver::pack(
  size_t s, bool from_next, bool from_current, double* x) const
{
  const auto& groups = group_indices_[s];
  const int64_t M = harmonics_.size(1);
  const int64_t G = num_groups_;
  for (const auto& P : patches_) {
    const int64_t N = P.disc->num_dofs();
    const auto& src = from_next ? P.moments_next : P.moments;
    const double* m = src.template data_ptr<double>();
    for (int64_t k = 0; k < M; k++)
      for (int64_t g : groups) {
        std::copy(m + (k * G + g) * N, m + (k * G + g + 1) * N, x);
        x += N;
      }
  }
  for (const auto& blk : delayed_) {
    const auto& buf = delayed_buffer(blk, from_current);
    const int64_t Nf = buf.size(2);
    const double* t = buf.template data_ptr<double>();
    for (int64_t d : blk.directions)
      for (int64_t g : groups) {
        std::copy(t + (d * G + g) * Nf, t + (d * G + g + 1) * Nf, x);
        x += Nf;
      }
  }
}

void SweepSolver::unpack(size_t s, const double* x, bool to_krylov)
{
  const auto& groups = group_indices_[s];
  const int64_t M = harmonics_.size(1);
  const int64_t G = num_groups_;
  for (auto& P : patches_) {
    const int64_t N = P.disc->num_dofs();
    auto& dst = to_krylov ? P.krylov_moments : P.moments;
    double* m = dst.template data_ptr<double>();
    for (int64_t k = 0; k < M; k++)
      for (int64_t g : groups) {
        std::copy(x, x + N, m + (k * G + g) * N);
        x += N;
      }
  }
  for (const auto& blk : delayed_) {
    auto& buf = delayed_buffer(blk, false);
    const int64_t Nf = buf.size(2);
    double* t = buf.template data_ptr<double>();
    for (int64_t d : blk.directions)
      for (int64_t g : groups) {
        std::copy(x, x + Nf, t + (d * G + g) * Nf);
        x += Nf;
      }
  }
}

void SweepSolver::report_progress(
  size_t s, int64_t iteration, double residual, const char* method, bool force)
{
  if (!config_.verbose || config_.progress_seconds < 0 || comm_.rank() != 0)
    return;
  auto now = std::chrono::steady_clock::now();
  if (!force && std::chrono::duration<double>(now - last_progress_).count() <
                  config_.progress_seconds)
    return;
  last_progress_ = now;
  const auto& groups = group_indices_[s];
  std::cout << "--   group set " << s + 1 << "/" << group_indices_.size()
            << " (groups " << groups.front() << "-" << groups.back() << "), "
            << method << " iteration " << iteration
            << ": residual = " << std::scientific << std::setprecision(3)
            << residual << std::defaultfloat << ", sweep time " << std::fixed
            << std::setprecision(1) << get_dag_seconds() << " s, solve time "
            << std::setprecision(1)
            << std::chrono::duration<double>(now - solve_start_).count() << " s"
            << std::defaultfloat << std::endl;
}

double SweepSolver::inner_tolerance() const
{
  const double floor =
    config_.inner_tol > 0 ? config_.inner_tol : 0.1 * config_.tol;
  if (config_.inner_forcing <= 0)
    return floor;
  const double change = errors_.empty() ? 1.0 : std::min(errors_.back(), 1.0);
  return std::max(floor, config_.inner_forcing * change);
}

double SweepSolver::global_dot(const std::vector<double>& a,
  const std::vector<double>& b, const linalg::krylov::Reducer& reducer)
{
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); i++)
    sum += a[i] * b[i];
  return reducer ? reducer(sum) : sum;
}

void SweepSolver::within_gmres(size_t s, bool jacobi,
  const std::vector<double>& rhs, std::vector<double>& x, double tol,
  double tol_min, double scale, const linalg::krylov::Reducer& reducer)
{
  // The unknown x is group set s's flux moments plus its lagged face traces
  // (cycle breaks, reflective faces): everything one transport solve reads
  // from the previous iterate. With T the transport solve (sweep or cross)
  // and S the within-set scattering, the fixed point x = T(S x) + b is
  // solved as (I - T S) x = rhs by GMRES. Standard form: rhs = b, the
  // transport solve of the fixed sources with zero iterate. Correction
  // form: x is the correction e and rhs the residual r0 = y0 - x0
  const int64_t n = static_cast<int64_t>(x.size());
  std::vector<double> tmp(n);

  // One product out = (I - T S) in: scatter `in` into the Krylov moments
  // and the lagged traces, build the source from the within-set
  // scattering of those moments only (fixed = false: no external, fission
  // or other-set terms, which are in rhs), transport, gather T S in
  auto apply = [&](const double* in, double* out) {
    timed("unpack", [&] { unpack(s, in, true); });
    timed("source", [&] { compute_source(s, Within::KRYLOV, false, jacobi); });
    timed("transport", [&] { transport_group_set(s); });
    timed("pack", [&] { pack(s, true, true, tmp.data()); });
    for (int64_t i = 0; i < n; i++)
      out[i] = in[i] - tmp[i];
  };

  // Called after every GMRES iteration with the relative residual
  // (relative to ||rhs||); true_residual marks one recomputed at a restart
  int64_t iteration = 0;
  double last_estimate = 1.0;
  auto progress = [&](double residual, bool true_residual) {
    if (!true_residual)
      last_estimate = residual;
    report_progress(s, ++iteration, last_estimate * scale,
      true_residual && iteration > 1 ? "GMRES (restart)" : "GMRES");
    // Relaxed inexact Krylov: the next product may be as inaccurate as
    // tol_min / (current residual), since its error is damped by the
    // residual. Products start at tol_min and loosen as GMRES converges,
    // capped at cross_max_tol. `scale` puts the residual on the norm the
    // tolerances are relative to (1, or ||r0|| / ||y0|| in correction
    // form). Only CROSS transport reads cross_tol_; an exact sweep ignores
    // it
    const double relative = residual * scale;
    cross_tol_ = config_.cross_relax ? std::min(config_.cross_max_tol,
                                         tol_min / std::max(relative, tol_min))
                                     : tol_min;
  };

  // Restarted GMRES to tol (relative to the scaled norm), x warm-started by
  // the caller; reducer sums inner products across ranks
  linalg::krylov::KrylovResult result;
  timed("gmres", [&] {
    result = linalg::krylov::gmres(apply, rhs, x, config_.gmres_restart,
      tol / scale, config_.inner_max_iter, reducer, progress);
  });
  // Later transport solves (e.g. the caller's final one) at full accuracy
  cross_tol_ = tol_min;
  if (!result.converged && config_.verbose && comm_.rank() == 0) {
    std::cout << "-- warning: within-group-set GMRES (set " << s
              << ") stopped at relative residual "
              << result.relative_residual * scale << std::endl;
  }
}

void SweepSolver::correction_solve(size_t s, bool jacobi, double tol,
  double tol_min, std::vector<double>& x,
  const linalg::krylov::Reducer& reducer)
{
  const int64_t n = static_cast<int64_t>(x.size());
  auto norm = [&](const std::vector<double>& v) {
    return std::sqrt(global_dot(v, v, reducer));
  };

  // y0 = T(q + S x0) with the lagged traces of x0, at full accuracy
  std::vector<double> y(n), r0(n);
  cross_tol_ = tol_min;
  timed("unpack", [&] { unpack(s, x.data(), true); });
  timed("source", [&] { compute_source(s, Within::KRYLOV, true, jacobi); });
  if (!try_incremental_transport(s, x, tol_min, y, reducer)) {
    if (config_.incremental_transport) {
      anchor_x_[s] = x;
      anchor_steps_[s] = 0;
      full_solves_++;
      swap_source(s, anchor_source_[s], false);
    }
    timed("transport", [&] { transport_group_set(s); });
    timed("pack", [&] { pack(s, true, true, y.data()); });
    if (config_.incremental_transport)
      anchor_y_[s] = y;
  }
  for (int64_t i = 0; i < n; i++)
    r0[i] = y[i] - x[i];
  const double ynorm = norm(y), rnorm = norm(r0);

  // (I - T S) e = r0, residuals relative to ||r0||, products accurate to
  // tol_min ||y0||
  std::vector<double> e(n, 0.0);
  if (rnorm > tol * ynorm)
    within_gmres(s, jacobi, r0, e, tol, tol_min, rnorm / ynorm, reducer);
  cross_tol_ = tol_min;
  if (rnorm > tol * ynorm) {
    for (int64_t i = 0; i < n; i++)
      x[i] += e[i];
  } else {
    // Already within tolerance: accept y0 = x0 + r0, which carries the
    // new sources (the standard form's final sweep)
    x = y;
  }

  // The stored angular flux and the kept TT come from the last transport
  // solve: redo it from the accepted iterate when they are wanted
  const bool keep_angular_flux = config_.transport == TransportSolve::CROSS &&
                                 config_.cross.keep_angular_flux;
  if (config_.store_angular_flux || keep_angular_flux) {
    unpack(s, x.data(), true);
    compute_source(s, Within::KRYLOV, true, jacobi);
    transport_group_set(s);
  }
  unpack(s, x.data(), false);
}

void SweepSolver::swap_source(
  size_t s, std::vector<double>& saved, bool difference)
{
  const auto& groups = group_indices_[s];
  const int64_t M = harmonics_.size(1);
  const int64_t G = num_groups_;
  int64_t size = 0;
  for (const auto& P : patches_)
    size += M * static_cast<int64_t>(groups.size()) * P.disc->num_dofs();
  if (difference && static_cast<int64_t>(saved.size()) != size)
    throw utils::runtime_error(
      ctx("swap_source"), "No previous source of this group set");
  saved.resize(size);
  double* prev = saved.data();
  for (auto& P : patches_) {
    const int64_t N = P.disc->num_dofs();
    double* src = P.source_moments.template data_ptr<double>();
    for (int64_t k = 0; k < M; k++)
      for (int64_t g : groups) {
        double* v = src + (k * G + g) * N;
        for (int64_t i = 0; i < N; i++) {
          const double cur = v[i];
          if (difference)
            v[i] = cur - prev[i];
          prev[i] = cur;
        }
        prev += N;
      }
  }
}

void SweepSolver::source_change(size_t s, const std::vector<double>& saved,
  double& change_sq, double& source_sq) const
{
  const auto& groups = group_indices_[s];
  const int64_t M = harmonics_.size(1);
  const double* prev = saved.data();
  for (const auto& P : patches_) {
    const int64_t N = P.disc->num_dofs();
    const double* src = P.source_moments.template data_ptr<double>();
    for (int64_t k = 0; k < M; k++)
      for (int64_t g : groups) {
        const double* v = src + (k * num_groups_ + g) * N;
        for (int64_t i = 0; i < N; i++) {
          const double d = v[i] - prev[i];
          change_sq += d * d;
          source_sq += v[i] * v[i];
        }
        prev += N;
      }
  }
}

bool SweepSolver::try_incremental_transport(size_t s,
  const std::vector<double>& x, double tol_min, std::vector<double>& y,
  const linalg::krylov::Reducer& reducer)
{
  if (!config_.incremental_transport || anchor_x_[s].empty())
    return false;
  // Each increment may add tol_min = inner_factor tol of error to y0:
  // a full solve once the drift could reach the inner tolerance
  if (static_cast<double>(anchor_steps_[s] + 1) * config_.cross_inner_factor >
      1.0)
    return false;
  const int64_t n = static_cast<int64_t>(x.size());
  const auto& ax = anchor_x_[s];
  auto& ay = anchor_y_[s];
  std::vector<double> dx(n);
  double sums[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int64_t i = 0; i < n; i++) {
    dx[i] = x[i] - ax[i];
    sums[0] += dx[i] * dx[i];
    sums[1] += x[i] * x[i];
    sums[2] += ay[i] * ay[i];
  }
  // The source moments' change (fission, other sets' scattering, fixed
  // source) since the anchor
  source_change(s, anchor_source_[s], sums[3], sums[4]);
  for (double& v : sums)
    v = reducer ? reducer(v) : v;
  const double dnorm = std::sqrt(sums[0]);
  const double sw = config_.incremental_switch;
  if (dnorm > sw * std::sqrt(sums[1]) ||
      std::sqrt(sums[3]) > sw * std::sqrt(sums[4]))
    return false;

  // The change, solved to the absolute accuracy tol_min ||y||
  timed("source", [&] { swap_source(s, anchor_source_[s], true); });
  timed("unpack", [&] { unpack(s, dx.data(), true); });
  cross_tol_ = dnorm > 0.0 ? std::min(config_.cross_max_tol,
                               tol_min * std::sqrt(sums[2]) / dnorm)
                           : config_.cross_max_tol;
  timed("transport", [&] { transport_group_set(s); });
  timed("pack", [&] { pack(s, true, true, y.data()); });
  cross_tol_ = tol_min;
  for (int64_t i = 0; i < n; i++)
    y[i] += ay[i];
  anchor_x_[s] = x;
  ay = y;
  anchor_steps_[s]++;
  incremental_solves_++;
  return true;
}

double SweepSolver::global_tolerance_scale(size_t s) const
{
  // Factor turning set s's inner tolerance (relative to the set's own flux)
  // into one relative to the whole flux (global_inner_tolerance): a set
  // carrying 1% of the flux only needs its error at tol of the whole, so it
  // may solve to about 100 tol of its own. Measured on the current
  // iterate's scalar flux (moment 0), in the outer test's coefficient norm
  const auto& groups = group_indices_[s];
  std::vector<uint8_t> in_set(num_groups_, 0);
  for (int64_t g : groups)
    in_set[g] = 1;
  // Squared flux norms over all groups and over the set, and the fission
  // production of all groups' flux and of the set's
  double all = 0.0, set = 0.0, prod_all = 0.0, prod_set = 0.0;
  for (const auto& P : patches_) {
    const int64_t N = P.disc->num_dofs();
    const double* m = P.moments.template data_ptr<double>();
    // The first G * N entries are moment 0 (the scalar flux)
    for (int64_t g = 0; g < num_groups_; g++) {
      double sum = 0.0;
      for (int64_t i = g * N; i < (g + 1) * N; i++)
        sum += m[i] * m[i];
      all += sum;
      if (in_set[g])
        set += sum;
    }
    if (P.fission.defined()) {
      // Fission production of all groups' flux, and of the set's
      const double* F = P.fission.template data_ptr<double>();
      const double* bi = P.disc->basis_integral().template data_ptr<double>();
      prod_all += transfer_total(P.disc->view(), F, num_groups_, m, bi);
      prod_set +=
        transfer_total(P.disc->view(), F, num_groups_, m, bi, in_set.data());
    }
  }
  all = global_sum(all);
  set = global_sum(set);
  prod_all = global_sum(prod_all);
  prod_set = global_sum(prod_set);
  // ||phi|| / ||phi_s|| (1 before the set has flux)
  double scale = set > 0.0 ? std::sqrt(all / set) : 1.0;
  // A set that makes neutrons stays as tight as its share of the fission
  // production asks, however small its flux: its error feeds every group
  // through the fission source. Sets without fission keep the flux ratio
  // (their error reaches the others only through scattering)
  if (prod_set > 0.0 && prod_all > 0.0)
    scale = std::min(scale, prod_all / prod_set);
  return scale;
}

int64_t SweepSolver::inner_solve(size_t s, bool jacobi)
{
  // Solves group set s for its within-set scattering with everything else
  // held fixed (external and fission sources, the other sets' scattering:
  // from the start of the pass with jacobi, else their latest flux). On
  // return the set's moments and traces are committed. Returns the
  // transport solves (group-set sweeps) used
  const int64_t start = group_set_sweeps_;

  // Inner tolerance: forced by the last outer change (inner_tolerance()),
  // optionally made relative to the whole flux and capped so a small set
  // still gets some accuracy
  double tol = inner_tolerance();
  if (config_.global_inner_tolerance)
    tol = std::min(GLOBAL_TOLERANCE_CAP, tol * global_tolerance_scale(s));
  // Cross accuracy: a fraction of the inner tolerance, relaxed in GMRES
  const double tol_min = config_.cross_inner_factor * tol;
  cross_tol_ = tol_min;

  // Richardson (source iteration): sweep with the latest moments, commit,
  // stop when the set's moments change by less than tol (relative)
  if (config_.inner_solver == InnerSolver::RICHARDSON) {
    const auto& groups = group_indices_[s];
    for (int64_t it = 0; it < config_.inner_max_iter; it++) {
      double num = 0.0, den = 0.0;
      compute_source(s, Within::MOMENTS, true, jacobi);
      transport_group_set(s);
      for (const auto& P : patches_) {
        const int64_t N = P.disc->num_dofs();
        const double* a = P.moments.template data_ptr<double>();
        const double* b = P.moments_next.template data_ptr<double>();
        for (int64_t g : groups)
          for (int64_t i = g * N; i < (g + 1) * N; i++) {
            num += (b[i] - a[i]) * (b[i] - a[i]);
            den += b[i] * b[i];
          }
      }
      commit_group_set(s);
      num = global_sum(num);
      den = global_sum(den);
      report_progress(
        s, it + 1, den == 0.0 ? 0.0 : std::sqrt(num / den), "Richardson");
      if (den == 0.0 || std::sqrt(num / den) < tol)
        break;
    }
    return group_set_sweeps_ - start;
  }

  // GMRES on (I - T S) x = b (see within_gmres), x the set's moments and
  // lagged traces
  const int64_t n = krylov_size(s);
  std::vector<double> x(n), b(n), zero(n, 0.0);
  pack(s, false, false, x.data()); // warm start from the current iterate
  const auto reducer =
    comm_.size() > 1
      ? linalg::krylov::Reducer([this](double v) { return global_sum(v); })
      : linalg::krylov::Reducer();
  if (rhs_ratio_.size() != group_indices_.size())
    rhs_ratio_.assign(group_indices_.size(), 0.0);
  // Correction form once a standard solve has measured ||b|| / ||x||: it
  // solves for the change from the warm start instead and writes the result
  // straight into the current moments and lagged traces (a final transport
  // solve only when psi or the kept TT is wanted). Tolerances relative to
  // ||b|| as in the standard form
  if (config_.correction_form && rhs_ratio_[s] > 0.0) {
    correction_solve(
      s, jacobi, tol * rhs_ratio_[s], tol_min * rhs_ratio_[s], x, reducer);
    return group_set_sweeps_ - start;
  }

  // b = sweep of the fixed sources with zero iterate and lagged traces
  timed("unpack", [&] { unpack(s, zero.data(), true); });
  timed("source", [&] { compute_source(s, Within::NONE, true, jacobi); });
  timed("transport", [&] { transport_group_set(s); });
  timed("pack", [&] { pack(s, true, true, b.data()); });

  within_gmres(s, jacobi, b, x, tol, tol_min, 1.0, reducer);
  // Record ||b|| / ||x|| so later correction-form solves of this set can
  // express their tolerances against ||b|| without forming b
  if (config_.correction_form) {
    const double bb = global_dot(b, b, reducer);
    const double xx = global_dot(x, x, reducer);
    rhs_ratio_[s] = xx > 0.0 ? std::sqrt(bb / xx) : 0.0;
  }

  // Consistent final sweep from the solution (traces, angular flux): the
  // GMRES products wrote Krylov vectors, not the state, so one more
  // transport solve (at tol_min) makes the moments, traces and psi agree
  // with x, and commits them
  unpack(s, x.data(), false);
  compute_source(s, Within::MOMENTS, true, jacobi);
  transport_group_set(s);
  commit_group_set(s);
  return group_set_sweeps_ - start;
}

int64_t SweepSolver::group_set_pass()
{
  const bool jacobi = config_.group_iteration == GroupIteration::JACOBI;
  if (jacobi)
    for_each_patch([](PatchState& P) { P.pass_moments.copy_(P.moments); });
  int64_t sweeps = 0;
  for (size_t s = 0; s < group_indices_.size(); s++)
    timed("inner", [&] { sweeps += inner_solve(s, jacobi); });
  return sweeps;
}

} // namespace ttnte::sweep
