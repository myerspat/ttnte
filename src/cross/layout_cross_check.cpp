// LayoutCross: The hold-out check and evaluation at scattered pairs.
#include "cross/layout_cross_detail.hpp"
#include "ttnte/cross/layout_cross.hpp"

namespace ttnte::cross {

using namespace detail;

double LayoutCross::check()
{
  const int64_t total = num_directions_ * size_[E];
  const int64_t holdout = std::max<int64_t>(config_.holdout, 1);
  const int64_t free = total - num_swept_;
  const bool fresh = free >= holdout;
  // Pool pairs the fit has since used (forced after a failed check, or
  // picked by selection) are in sample now: drop them from the pool
  pool_.erase(std::remove_if(pool_.begin(), pool_.end(),
                [&](int64_t p) { return fitted_[p] != 0; }),
    pool_.end());
  if (fresh) {
    std::vector<int64_t> pick;
    if (free > total / 4) {
      std::uniform_int_distribution<int64_t> dist(0, total - 1);
      std::set<int64_t> chosen;
      while (static_cast<int64_t>(chosen.size()) < holdout) {
        const int64_t p = dist(rng_);
        if (!swept(p))
          chosen.insert(p);
      }
      pick.assign(chosen.begin(), chosen.end());
    } else {
      std::vector<int64_t> all;
      for (int64_t p = 0; p < total; p++)
        if (!swept(p))
          all.push_back(p);
      std::sample(
        all.begin(), all.end(), std::back_inserter(pick), holdout, rng_);
    }
    ensure(pick, "check");
    pool_.insert(pool_.end(), pick.begin(), pick.end());
  } else {
    // (Nearly) everything swept: judge on fresh random swept fibers, those
    // the fit has not used if there are enough of them
    std::vector<int64_t> all;
    for (int64_t p = 0; p < total; p++)
      if (swept(p) && !fitted_[p])
        all.push_back(p);
    if (static_cast<int64_t>(all.size()) < holdout) {
      all.clear();
      for (int64_t p = 0; p < total; p++)
        if (swept(p))
          all.push_back(p);
    }
    pool_.clear();
    std::sample(all.begin(), all.end(), std::back_inserter(pool_),
      config_.resampled_pool * holdout, rng_);
  }
  if (pool_.empty()) {
    pool_error_ = global_error_ = 0.0;
    return 0.0;
  }

  // Squared errors of the TT at swept pairs, and their total squared norm.
  // Plain loops: these arrays are small, and tensor reductions on them cost
  // more in thread start-up than in arithmetic
  const int64_t Ns = size_[S];
  auto errors = [&](const std::vector<int64_t>& pairs) {
    const int64_t n = static_cast<int64_t>(pairs.size());
    const auto approx = evaluate_pairs(pairs);
    const double* a = approx.data_ptr<double>();
    std::vector<double> err(n, 0.0);
    double norm_sq = 0.0;
    for (int64_t b = 0; b < n; b++) {
      const double* f = fiber(pairs[b]);
      double e = 0.0, x = 0.0;
      for (int64_t s = 0; s < Ns; s++) {
        const double d = a[b * Ns + s] - f[s];
        e += d * d;
        x += f[s] * f[s];
      }
      err[b] = e;
      norm_sq += x;
    }
    return std::make_pair(err, norm_sq);
  };
  const int64_t B = static_cast<int64_t>(pool_.size());
  auto [err_b, pool_sq] = errors(pool_);
  const double pool_err_sq = std::accumulate(err_b.begin(), err_b.end(), 0.0);
  holdout_rms_ = std::sqrt(pool_sq / static_cast<double>(B));
  pool_error_ = std::sqrt(pool_err_sq / std::max(pool_sq, 1e-300));

  // Cheap first stage: a pool within the acceptance accepts the pass, and
  // the global estimate is not formed (NaN). Otherwise, with global_holdout,
  // the pass may still be accepted on the global estimate: the pool judges
  // its fibers against their own norm, and fails on small fibers the train
  // resolves relative to the whole flux
  const double accept = config_.holdout_factor * config_.tol;
  global_error_ = std::numeric_limits<double>::quiet_NaN();
  auto force_worst = [&] {
    // The worst pool pairs (absolute error) go into every index set next
    // pass: only pool fibers are out of sample
    std::vector<int64_t> order(B);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
      [&](int64_t x, int64_t y) { return err_b[x] > err_b[y]; });
    for (int64_t i = 0; i < std::min<int64_t>(config_.forced_per_check, B);
      i++) {
      const int64_t p = pool_[order[i]];
      if (std::find(force_.begin(), force_.end(), p) == force_.end())
        force_.push_back(p);
    }
  };
  if (pool_error_ <= accept)
    return pool_error_;
  if (!config_.global_holdout) {
    force_worst();
    return pool_error_;
  }

  // Global estimate, by strata. Out of sample: the pairs unswept before
  // this check plus the older pool fibers, estimated by the pool. In sample:
  // the other swept fibers (norms exact), their errors estimated from a
  // random sample. Once nearly everything is swept the pool is itself a
  // sample of swept fibers and estimates every pair.
  double err_sq = 0.0, norm_sq = 0.0;
  if (fresh) {
    const double out = static_cast<double>(free + B - holdout);
    const double in = static_cast<double>(num_swept_ - B);
    norm_sq = (swept_sq_ - pool_sq) + out * pool_sq / B;
    err_sq = out * pool_err_sq / B;
    if (in > 0) {
      std::unordered_set<int64_t> in_pool(pool_.begin(), pool_.end());
      std::vector<int64_t> candidates;
      candidates.reserve(num_swept_ - B);
      for (int64_t p = 0; p < total; p++)
        if (swept(p) && !in_pool.count(p))
          candidates.push_back(p);
      std::vector<int64_t> sample;
      std::sample(candidates.begin(), candidates.end(),
        std::back_inserter(sample), config_.swept_sample * holdout,
        sample_rng_);
      if (!sample.empty()) {
        auto [in_err, in_norm] = errors(sample);
        err_sq += in * std::accumulate(in_err.begin(), in_err.end(), 0.0) /
                  static_cast<double>(sample.size());
      }
    }
  } else {
    const double unswept = static_cast<double>(free);
    norm_sq = swept_sq_ + unswept * swept_sq_ / std::max<double>(num_swept_, 1);
    err_sq = static_cast<double>(total) * pool_err_sq / B;
  }
  global_error_ = std::sqrt(err_sq / std::max(norm_sq, 1e-300));
  if (global_error_ > accept)
    force_worst();
  return global_error_;
}

torch::Tensor LayoutCross::evaluate_pairs(
  const std::vector<int64_t>& pairs) const
{
  if (!has_tt())
    throw utils::runtime_error(ctx("evaluate_pairs"),
      "No TT held: run() first, or set "
      "keep_angular_flux to keep it after a solve");
  const int64_t n = static_cast<int64_t>(pairs.size());
  const int64_t G = size_[E], Ng = size_[GA], Ns = size_[S];
  auto out = torch::zeros({n, Ns}, torch::kFloat64);
  if (n == 0)
    return out;
  std::vector<std::array<int64_t, 4>> comp(n);
  for (int64_t b = 0; b < n; b++) {
    if (pairs[b] < 0 || pairs[b] >= num_directions_ * G)
      throw utils::runtime_error(ctx("evaluate_pairs"), "Pair out of range");
    const int64_t d = pairs[b] / G;
    comp[b] = {d / Ng, d % Ng, -1, pairs[b] % G};
  }
  // Index of each pair's values in core k (its non-space modes; the space
  // core's index is taken as outer * inner + inner index, without space)
  int64_t ks = -1;
  for (int64_t k = 0; k < K_; k++)
    for (int m : core_modes_[k])
      if (m == S)
        ks = k;
  auto core_index = [&](int64_t k, bool space_core) {
    std::vector<int64_t> idx(n);
    for (int64_t b = 0; b < n; b++) {
      int64_t j = 0;
      for (size_t q = 0; q < core_modes_[k].size(); q++) {
        const int m = core_modes_[k][q];
        if (m == S && space_core)
          continue;
        j = j * core_dims_[k][q] + comp[b][m];
      }
      idx[b] = j;
    }
    return idx;
  };
  // Left factors (n, r_ks) from the cores before space, right factors
  // (n, r_ks+1) from the cores after it
  auto L = torch::ones({n, 1}, torch::kFloat64);
  for (int64_t k = 0; k < ks; k++) {
    auto J = torch::tensor(core_index(k, false), torch::kInt64);
    L = torch::einsum("na,anb->nb", {L, cores_[k].index_select(1, J)});
  }
  auto R = torch::ones({n, 1}, torch::kFloat64);
  for (int64_t k = K_ - 1; k > ks; k--) {
    auto J = torch::tensor(core_index(k, false), torch::kInt64);
    R = torch::einsum("anb,nb->na", {cores_[k].index_select(1, J), R});
  }
  L = L.contiguous();
  R = R.contiguous();
  // The space core as (r_l, outer, Ns, inner, r_r): pairs grouped by their
  // (outer, inner) index read each slice once, in one product per group
  const auto [outer, inner] = space_split(ks);
  const auto& c = cores_[ks];
  auto core5 = c.view({c.size(0), outer, Ns, inner, c.size(2)});
  const auto key = core_index(ks, true);
  std::map<int64_t, std::vector<int64_t>> groups;
  for (int64_t b = 0; b < n; b++)
    groups[key[b]].push_back(b);
  std::vector<std::pair<int64_t, std::vector<int64_t>>> list(
    groups.begin(), groups.end());
  double* o = out.data_ptr<double>();
  at::parallel_for(
    0, static_cast<int64_t>(list.size()), 1, [&](int64_t g0, int64_t g1) {
      for (int64_t g = g0; g < g1; g++) {
        const auto& [k, members] = list[g];
        auto idx = torch::tensor(members, torch::kInt64);
        auto slice =
          core5.select(1, k / inner).select(2, k % inner); // (r_l, Ns, r_r)
        auto tmp =
          torch::matmul(slice, R.index_select(0, idx).t()); // (r_l, Ns, nb)
        auto vals = torch::einsum("asb,ba->bs", {tmp, L.index_select(0, idx)})
                      .contiguous();
        const double* v = vals.data_ptr<double>();
        for (size_t i = 0; i < members.size(); i++)
          std::copy(v + i * Ns, v + (i + 1) * Ns, o + members[i] * Ns);
      }
    });
  return out;
}

} // namespace ttnte::cross
