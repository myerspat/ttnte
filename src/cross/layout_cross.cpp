// LayoutCross: Construction, bookkeeping and the forward / backward passes.
#include "ttnte/cross/layout_cross.hpp"
#include "cross/layout_cross_detail.hpp"

namespace ttnte::cross {

using namespace detail;

// =================================================================
// Construction and bookkeeping

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double>
LayoutCross::svd(const torch::Tensor& A, double rel_tol) const
{
  return config_.gram_svd
           // The cores are fitted by least squares on selected rows, so the
           // long factor need not be re-orthogonalized
           ? linalg::gram_truncated_svd(A, rel_tol, max_rank(), true,
               config_.gram_floor, /*orthonormal=*/false)
           : linalg::truncated_svd(A, rel_tol, max_rank(), true);
}

int64_t LayoutCross::max_rank() const noexcept
{
  return config_.max_rank > 0 ? config_.max_rank
                              : std::numeric_limits<int64_t>::max();
}

LayoutCross::LayoutCross(CrossProblem problem, CrossConfig config)
  : problem_(std::move(problem)), config_(std::move(config)),
    rng_(config_.seed), sample_rng_(config_.seed ^ SAMPLE_SEED_MIX)
{
  const auto& P = problem_;
  if (P.num_dim < 1 || P.num_dim > 3 || P.num_polar < 1 ||
      P.num_azimuthal < 1 || P.num_groups < 1 || P.num_space < 1) {
    throw utils::runtime_error(ctx("LayoutCross"), "Invalid problem sizes");
  }
  if (!P.sweep)
    throw utils::runtime_error(ctx("LayoutCross"), "No sweep oracle");
  num_directions_ = P.num_polar * P.num_azimuthal;
  if (!P.weights.defined() || P.weights.numel() != num_directions_ ||
      !P.harmonics.defined() || P.harmonics.size(0) != num_directions_) {
    throw utils::runtime_error(ctx("LayoutCross"),
      "weights and harmonics must have one row per direction");
  }
  // The fiber layout: weighting writes through raw pointers at these
  // offsets, so they must tile the fiber exactly
  if (!P.patch_offsets.empty()) {
    if (P.patch_offsets.size() != P.patch_shapes.size() ||
        P.patch_offsets.front() != 0) {
      throw utils::runtime_error(ctx("LayoutCross"),
        "patch_offsets must start at 0 with one shape per patch");
    }
    for (size_t q = 0; q < P.patch_offsets.size(); q++) {
      const int64_t end =
        q + 1 < P.patch_offsets.size() ? P.patch_offsets[q + 1] : P.num_space;
      int64_t n = 1;
      for (int64_t d : P.patch_shapes[q])
        n *= d;
      if (end - P.patch_offsets[q] != n) {
        throw utils::runtime_error(
          ctx("LayoutCross"), "Patch " + std::to_string(q) +
                                ": offsets and control-point shape disagree");
      }
      if (!P.mass_bands.empty() &&
          (P.mass_bands[q].dim() != 2 || P.mass_bands[q].size(0) != n)) {
        throw utils::runtime_error(
          ctx("LayoutCross"), "Patch " + std::to_string(q) +
                                ": its mass band needs one row per dof");
      }
    }
  }
  // init_right() sweeps every group of these to choose the initial energies
  // and right sets; there is no sensible default
  if (P.initial_directions.empty())
    throw utils::runtime_error(ctx("LayoutCross"),
      "initial_directions must name at least one direction (the sweeper "
      "uses one per quadrant or octant)");
  for (int64_t d : P.initial_directions) {
    if (d < 0 || d >= num_directions_)
      throw utils::runtime_error(
        ctx("LayoutCross"), "Initial direction out of range");
  }
  if (config_.tol <= 0)
    throw utils::runtime_error(ctx("LayoutCross"), "tol must be positive");
  if (!(config_.select_cond > 0 && config_.select_cond < 1))
    throw utils::runtime_error(
      ctx("LayoutCross"), "select_cond must be in (0, 1)");
  if (!(config_.select_near > 0 && config_.select_near <= 1))
    throw utils::runtime_error(
      ctx("LayoutCross"), "select_near must be in (0, 1]");
  if (config_.stall_passes < 0)
    throw utils::runtime_error(
      ctx("LayoutCross"), "stall_passes must be nonnegative");
  if (!(config_.stall_improvement >= 0 && config_.stall_improvement < 1))
    throw utils::runtime_error(
      ctx("LayoutCross"), "stall_improvement must be in [0, 1)");
  auto in_unit = [](double x, bool closed) {
    return x > 0.0 && (closed ? x <= 1.0 : x < 1.0);
  };
  if (!in_unit(config_.forward_truncation_share, true) ||
      !in_unit(config_.growth_share, true))
    throw utils::runtime_error(ctx("LayoutCross"),
      "forward_truncation_share and growth_share must be in (0, 1]");
  if (!in_unit(config_.gram_solve_min_ratio, false) ||
      !in_unit(config_.gram_floor, false))
    throw utils::runtime_error(ctx("LayoutCross"),
      "gram_solve_min_ratio and gram_floor must be in (0, 1)");
  if (config_.resampled_pool < 1 || config_.swept_sample < 1 ||
      config_.forced_per_check < 1 || config_.growth_max_blocks < 1 ||
      config_.sketch_rows < 1)
    throw utils::runtime_error(ctx("LayoutCross"),
      "resampled_pool, swept_sample, forced_per_check, growth_max_blocks and "
      "sketch_rows must be at least 1");

  user_layout_ = config_.layout.num_cores() > 0
                   ? config_.layout
                   : tensor::Layout::canonical(P.num_dim, false);
  user_layout_.validate(P.num_dim);
  layout_ = user_layout_.merge_space();
  K_ = static_cast<int64_t>(layout_.num_cores());
  size_ = {P.num_polar, P.num_azimuthal, P.num_space, P.num_groups};
  for (const auto& core : layout_.cores()) {
    std::vector<int> modes;
    std::vector<int64_t> dims;
    int64_t n = 1;
    for (const auto& idx : core) {
      const int mode = mode_of(idx.dimension());
      modes.push_back(mode);
      dims.push_back(size_[mode]);
      n *= size_[mode];
    }
    core_modes_.push_back(std::move(modes));
    core_dims_.push_back(std::move(dims));
    n_.push_back(n);
  }

  // Weights of the samples
  auto w = P.weights.to(torch::kFloat64).contiguous();
  angle_weight_.assign(num_directions_, 1.0);
  energy_weight_.assign(P.num_groups, 1.0);
  auto [wu, wv] =
    outer_factors(w.reshape({P.num_polar, P.num_azimuthal}), 1e-10);
  polar_weight_ = wu;
  azimuthal_weight_ = wv;
  if (config_.weighted) {
    const double* wp = w.data_ptr<double>();
    for (int64_t d = 0; d < num_directions_; d++)
      angle_weight_[d] = std::sqrt(wp[d]);
    const bool split =
      P.num_dim > 1 && layout_.core_of(tensor::Dimension::POLAR) !=
                         layout_.core_of(tensor::Dimension::AZIMUTHAL);
    if (split && !polar_weight_.defined()) {
      throw utils::runtime_error(ctx("LayoutCross"),
        "Weighting a layout that splits polar from azimuthal angle needs a "
        "product quadrature");
    }
    if (!P.lethargy_widths.empty()) {
      if (static_cast<int64_t>(P.lethargy_widths.size()) != P.num_groups) {
        throw utils::runtime_error(
          ctx("LayoutCross"), "lethargy_widths must have one entry per group");
      }
      for (int64_t g = 0; g < P.num_groups; g++) {
        if (!(P.lethargy_widths[g] > 0.0))
          throw utils::runtime_error(
            ctx("LayoutCross"), "lethargy_widths must be positive");
        energy_weight_[g] = 1.0 / std::sqrt(P.lethargy_widths[g]);
      }
    }
    for (const auto& band : P.mass_bands) {
      auto factor = band.to(torch::kFloat64).clone().contiguous();
      const int64_t kd = factor.size(1) - 1;
      if (linalg::banded::cholesky_factor(
            factor.data_ptr<double>(), factor.size(0), kd) != 0) {
        throw utils::runtime_error(
          ctx("LayoutCross"), "A patch mass matrix is not positive definite");
      }
      mass_factors_.push_back(factor);
      mass_kd_.push_back(kd);
      // Dense R = L^T from the lower band, when it fits (BLAS-3 batches)
      const int64_t n = factor.size(0);
      torch::Tensor dense;
      if (n <= DENSE_MASS_MAX) {
        dense = torch::zeros({n, n}, torch::kFloat64);
        const double* f = factor.data_ptr<double>();
        double* R = dense.data_ptr<double>();
        for (int64_t j = 0; j < n; j++)
          for (int64_t i = j; i <= std::min(n - 1, j + kd); i++)
            R[j * n + i] = f[j * (kd + 1) + (i - j)];
      }
      mass_dense_.push_back(dense);
    }
    if (!mass_factors_.empty() &&
        mass_factors_.size() != P.patch_offsets.size()) {
      throw utils::runtime_error(
        ctx("LayoutCross"), "One mass band per patch is required");
    }
  }
}

void LayoutCross::reset()
{
  fiber_blocks_.clear();
  slot_.assign(num_directions_ * size_[E], nullptr);
  fitted_.assign(num_directions_ * size_[E], 0);
  num_swept_ = 0;
  swept_sq_ = 0.0;
  holdout_rms_ = 0.0;
  force_.clear();
  grow_.clear();
  pool_.clear();
  passes_.clear();
  sweep_seconds_ = 0.0;
  IndexSet unset;
  unset.defined = false;
  IndexSet empty;
  empty.comp = {-1, -1, -1, -1};
  left_.assign(K_ + 1, unset);
  right_.assign(K_ + 1, unset);
  left_[0] = empty;
  right_[K_] = empty;
  B_.assign(K_ + 1, {});
  PL_.assign(K_ + 1, {});
  C_.assign(K_ + 1, {});
  PR_.assign(K_ + 1, {});
  cores_.assign(K_, {});
}

int64_t LayoutCross::key(int64_t mu, int64_t ga, int64_t e) const noexcept
{
  return ((mu + 1) * (size_[GA] + 1) + (ga + 1)) * (size_[E] + 1) + (e + 1);
}

std::pair<int64_t, int64_t> LayoutCross::space_split(size_t k) const
{
  const auto& modes = core_modes_[k];
  const auto qs = std::find(modes.begin(), modes.end(), static_cast<int>(S));
  if (qs == modes.end())
    return {0, 0};
  int64_t inner = 1;
  for (size_t q = (qs - modes.begin()) + 1; q < modes.size(); q++)
    inner *= core_dims_[k][q];
  return {n_[k] / (size_[S] * inner), inner};
}

int64_t LayoutCross::row_key(const int64_t* comp) const noexcept
{
  return key(comp[MU], comp[GA], comp[E]);
}

void LayoutCross::decode(size_t k, int64_t j, int64_t* comp) const
{
  const auto& modes = core_modes_[k];
  const auto& dims = core_dims_[k];
  for (int64_t q = static_cast<int64_t>(modes.size()) - 1; q >= 0; q--) {
    comp[modes[q]] = j % dims[q];
    j /= dims[q];
  }
}

int64_t LayoutCross::pair_of(const int64_t* comp) const
{
  const int64_t d = comp[MU] * size_[GA] + std::max<int64_t>(comp[GA], 0);
  return d * size_[E] + comp[E];
}

// =================================================================
// The cross

void LayoutCross::init_right()
{
  const int64_t G = size_[E], Ns = size_[S], Ng = size_[GA];
  std::vector<int64_t> D0;
  for (int64_t d : problem_.initial_directions)
    if (std::find(D0.begin(), D0.end(), d) == D0.end())
      D0.push_back(d);
  std::vector<int64_t> pairs;
  for (int64_t d : D0)
    for (int64_t g = 0; g < G; g++)
      pairs.push_back(d * G + g);
  ensure(pairs, "init");

  // Initial energies: pivots of a Gaussian sketch of the swept rows
  // The swept rows R0 ((|D0| Ns, G)) once, and one product with the
  // concatenated per-direction sketches (cached: they depend only on the
  // directions and the sketch size): a few small products at many threads
  // cost more in overhead than in arithmetic
  const int64_t nd = static_cast<int64_t>(D0.size());
  auto R0 = torch::empty({nd * Ns, G}, torch::kFloat64);
  {
    double* r = R0.data_ptr<double>();
    for (int64_t i = 0; i < nd; i++)
      for (int64_t g = 0; g < G; g++) {
        const double* f = fiber(D0[i] * G + g);
        for (int64_t s = 0; s < Ns; s++)
          r[(i * Ns + s) * G + g] = f[s];
      }
  }
  std::vector<int64_t> J0;
  for (int64_t kk = config_.sketch_rows;; kk *= 2) {
    auto& omega = sketches_[{nd, kk}];
    if (!omega.defined() || omega.size(1) != nd * Ns) {
      std::vector<torch::Tensor> parts;
      for (int64_t d : D0)
        parts.push_back(direction_sketch(d, kk, Ns));
      omega = torch::cat(parts, 1);
    }
    auto sketch = omega.mm(R0);
    J0 = pivot_columns(sketch, config_.tol);
    if (static_cast<int64_t>(J0.size()) <= kk / 2 ||
        kk >= std::min<int64_t>(G, static_cast<int64_t>(D0.size()) * Ns))
      break;
  }
  // Flags of the initial energies, directions and their angles (array
  // lookups: the loop below visits every core value)
  std::vector<uint8_t> J0s(G, 0), D0s(num_directions_, 0), mu0(size_[MU], 0),
    ga0(Ng, 0);
  for (int64_t g : J0)
    J0s[g] = 1;
  for (int64_t d : D0) {
    D0s[d] = 1;
    mu0[d / Ng] = 1;
    ga0[d % Ng] = 1;
  }

  for (int64_t k = K_ - 1; k >= 1; k--) {
    const auto& R = right_[k + 1];
    const int64_t mr = R.size();
    bool has[4] = {false, false, false, false};
    for (int64_t c = k; c < K_; c++)
      for (int m : core_modes_[c])
        has[m] = true;
    std::vector<int64_t> rows;
    for (int64_t j = 0; j < n_[k]; j++) {
      int64_t cj[4] = {-1, -1, -1, -1};
      decode(k, j, cj);
      for (int64_t l = 0; l < mr; l++) {
        int64_t comp[4];
        std::copy(&R.comp[l * 4], &R.comp[l * 4] + 4, comp);
        merge(comp, cj);
        bool ok = true;
        if (has[E])
          ok = ok && J0s[comp[E]];
        if (has[MU] && has[GA])
          ok = ok && D0s[comp[MU] * Ng + comp[GA]];
        else if (has[MU])
          ok = ok && mu0[comp[MU]];
        else if (has[GA])
          ok = ok && ga0[comp[GA]];
        if (ok)
          rows.push_back(j * mr + l);
      }
    }
    if (rows.empty()) {
      auto keys = candidate_keys_right(k);
      std::vector<int64_t> uk(keys);
      std::sort(uk.begin(), uk.end());
      uk.erase(std::unique(uk.begin(), uk.end()), uk.end());
      std::shuffle(uk.begin(), uk.end(), rng_);
      uk.resize(std::min<size_t>(4, uk.size()));
      for (int64_t c = 0; c < static_cast<int64_t>(keys.size()); c++)
        if (std::find(uk.begin(), uk.end(), keys[c]) != uk.end())
          rows.push_back(c);
    }
    right_[k] = candidates(k, false, rows);
    C_[k] = torch::Tensor();
    PR_[k] = torch::Tensor();
  }
}

void LayoutCross::forward()
{
  // The right sets are fixed for this half sweep and used as interpolation
  // on their samples (no P_R), so each bond's rank may grow up to its right
  // set's size
  for (int64_t k = 0; k + 1 < K_; k++) {
    C_[k + 1] = torch::Tensor();
    PR_[k + 1] = torch::Tensor();
  }

  // Per-bond truncation tolerance: forward_truncation_share (default 0.1)
  // of the backward pass's tol / sqrt(K - 1). The backward pass sets the
  // final ranks
  const double share =
    config_.forward_truncation_share / std::sqrt(std::max<double>(K_ - 1, 1));

  for (int64_t k = 0; k + 1 < K_; k++) {
    // Core k: the left factor of the left-projected superblock
    // P_L S_k, (r_{k-1}, n_k, |right_[k + 1]|), unfolded as
    // (r_{k-1} n_k) x |right_[k + 1]| and truncated
    auto Sb = superblock(k, false);
    auto t0 = Clock::now();
    const int64_t rl = Sb.size(0), nk = Sb.size(1), rr = Sb.size(2);
    const auto U =
      std::get<0>(svd(Sb.reshape({rl * nk, rr}), share * config_.tol));
    const int64_t r = U.size(1);
    auto core = U.reshape({rl, nk, r}).contiguous();
    cores_[k] = core;
    timings_["svd"] += elapsed(t0);

    // Next left set, chosen from left_[k] x core k's values, and its basis
    // B_{k+1} (rows B_k[a] G_k(j) at the chosen rows). The previous set's
    // blocks are kept: index sets only grow within a run
    t0 = Clock::now();
    std::vector<int64_t> keep;
    if (left_[k + 1].defined)
      keep = set_keys(left_[k + 1]);
    auto [rows, basis, gram] = select_candidates(k, true, core, keep);
    left_[k + 1] = candidates(k, true, rows);
    B_[k + 1] = basis;
    timings_["select"] += elapsed(t0);

    // P_L = B_{k+1}^+, which projects the next superblock's left side
    t0 = Clock::now();
    PL_[k + 1] = left_inverse(B_[k + 1], gram, config_.gram_solve_min_ratio);
    timings_["basis"] += elapsed(t0);
  }

  // Last core: the least-squares fit B_{K-1}^+ S_K on its raw superblock
  // (|left_[K - 1]|, n_{K-1}, 1). Only here does its misfit grow the bond:
  // the backward pass refits every core from the right sets this growth
  // enlarged, and the check forces pairs where the fit is poor
  auto Sraw = superblock(K_ - 1, true);
  auto t0 = Clock::now();
  auto core = PL_[K_ - 1].defined()
                ? torch::tensordot(PL_[K_ - 1], Sraw, {1}, {0})
                : Sraw;
  cores_[K_ - 1] = core.contiguous();
  if (K_ > 1) {
    // Misfit energy per value of the last core, summed over the left rows
    auto fit = B_[K_ - 1].defined()
                 ? torch::tensordot(B_[K_ - 1], core, {1}, {0})
                 : core;
    auto col = (Sraw - fit).select(2, 0).square().sum(0);
    grow_from_residual(col, Sraw.norm().item<double>(), K_ - 1,
      core_mask(K_ - 1, K_ - 1), right_[K_ - 1]);
  }
  timings_["svd"] += elapsed(t0);

  if (record_)
    passes_.push_back({true, cores_, B_, left_, right_});
}

void LayoutCross::backward()
{
  // Per-bond truncation tolerance tol / sqrt(K - 1), as in TT-SVD. The
  // truncation is relative to the projected superblock P_L A P_R, so in
  // fiber space its error can be amplified by up to cond(B) cond(C) <=
  // 1 / select_cond^2 (selection's target); the hold-out check, on the
  // fibers themselves, is what accepts the result
  const double share = 1.0 / std::sqrt(std::max<double>(K_ - 1, 1));

  for (int64_t k = K_ - 1; k >= 1; k--) {
    // Core k: the right factor of the superblock projected on both sides,
    // P_L S_k P_R, (r_{k-1}, n_k, r_k), unfolded as r_{k-1} x (n_k r_k)
    // and truncated. P_R comes from bond k + 1, chosen just before
    auto Sb = superblock(k, false);
    auto t0 = Clock::now();
    const int64_t rl = Sb.size(0), nk = Sb.size(1), rr = Sb.size(2);
    const auto Vh =
      std::get<2>(svd(Sb.reshape({rl, nk * rr}), share * config_.tol));
    const int64_t r = Vh.size(0);
    auto core = Vh.reshape({r, nk, rr}).contiguous();
    cores_[k] = core;
    timings_["svd"] += elapsed(t0);

    // Next right set, chosen from core k's values x right_[k + 1] (keeping
    // the current set's blocks, the residual-grown ones and the forced
    // pairs), and its basis C_k (columns G_k(j) C_{k+1}[:, l], stored as
    // rows)
    t0 = Clock::now();
    auto [rows, basis, gram] =
      select_candidates(k, false, core, set_keys(right_[k]));
    right_[k] = candidates(k, false, rows);
    C_[k] = basis;
    timings_["select"] += elapsed(t0);

    // P_R = (C_k^+)^T, which projects the next superblock's right side
    t0 = Clock::now();
    PR_[k] = left_inverse(basis, gram, config_.gram_solve_min_ratio).t();
    timings_["basis"] += elapsed(t0);
  }

  // First core: the least-squares fit S_1 C_2^+ on its raw superblock
  // (1, n_0, |right_[1]|)
  auto Sraw = superblock(0, true);
  auto t0 = Clock::now();
  cores_[0] =
    (PR_[1].defined() ? torch::tensordot(Sraw, PR_[1], {2}, {0}) : Sraw)
      .contiguous();
  timings_["svd"] += elapsed(t0);

  if (record_)
    passes_.push_back({false, cores_, C_, left_, right_});
}

void LayoutCross::run()
{
  auto t0 = Clock::now();

  // Fresh run: empty fiber store, index sets, forced pairs, growth and
  // hold-out pool (reset()), and no pass history
  reset();
  history_.clear();

  // init_right() and check() sweep fibers themselves. Their phase timings
  // exclude that sweeping, which is counted under "sweep" (and includes
  // the weighting of the new fibers): `sampling` reads that counter before
  // and after, and the difference is subtracted
  auto sampling = [this] { return timings_["sweep"]; };

  // Initial right index sets: the previous run's (warm start, so the
  // fibers it swept are sampled again), else built from the initial
  // directions and the energies pivoted from a Gaussian sketch of their
  // fibers (init_right())
  auto t1 = Clock::now();
  double s1 = sampling();
  if (config_.warm_start && !prior_right_.empty()) {
    for (int64_t k = 1; k < K_; k++)
      right_[k] = prior_right_[k];
  } else {
    init_right();
  }
  timings_["init"] += elapsed(t1) - (sampling() - s1);

  // A pass is accepted once the hold-out error is at most this
  const double accept = config_.holdout_factor * config_.tol;
  double err = std::numeric_limits<double>::infinity();
  sweeps_done_ = 0;
  stalled_ = false;

  // Stall detection: the ranks and error of the previous pass, and how many
  // passes in a row have changed neither enough
  std::vector<int64_t> prev_ranks;
  double prev_err = err;
  int64_t stall = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Passes (Alg. crossK): forward half sweep, backward half sweep, hold-out
  // check; at least one pass even if max_sweeps < 1
  for (int64_t it = 0; it < std::max<int64_t>(config_.max_sweeps, 1); it++) {
    // Forward: cores 0 ... K - 2 and the left index sets, then the last
    // core fitted by least squares and the last bond grown from its misfit.
    // The forward record has no check, so its error fields are NaN
    forward();
    history_.push_back({true, ranks(), num_swept_, nan, nan, nan, nan,
      static_cast<int64_t>(force_.size())});

    // Backward: cores K - 1 ... 1 truncated and the right index sets
    // chosen, then core 0 fitted by least squares
    backward();
    sweeps_done_ = it + 1;

    // Hold-out check of the new train on fibers it was not fitted to. A
    // failed check forces its worst pairs into the next pass's index sets
    t1 = Clock::now();
    s1 = sampling();
    err = check();
    timings_["check"] += elapsed(t1) - (sampling() - s1);

    // Backward record: the ranks, the check's errors and norms, and the
    // pairs forced so far
    const auto r = ranks();
    history_.push_back({false, r, num_swept_, pool_error_, holdout_rms_,
      std::sqrt(swept_sq_ / std::max<double>(num_swept_, 1)), global_error_,
      static_cast<int64_t>(force_.size())});

    if (err <= accept)
      break;

    // Stalled: ranks unchanged and the error hardly lower than last pass,
    // for stall_passes passes in a row (0 disables the test)
    const bool still =
      r == prev_ranks && err > (1.0 - config_.stall_improvement) * prev_err;
    stall = still ? stall + 1 : 0;
    if (config_.stall_passes > 0 && stall >= config_.stall_passes) {
      stalled_ = true;
      break;
    }
    prev_ranks = r;
    prev_err = err;
  }

  // Results of the run: the error the last check accepted or rejected on
  holdout_error_ = err;
  converged_ = err <= accept;
  if (config_.warm_start)
    prior_right_ = right_;

  // Remove the angle, energy and space weights from the cores, so the
  // train approximates the angular flux itself
  t1 = Clock::now();
  unweight();
  timings_["unweight"] += elapsed(t1);

  // Release the fiber store: the fibers are only needed during the run
  total_fibers_ += num_swept_;
  fiber_blocks_.clear();
  std::fill(slot_.begin(), slot_.end(), nullptr);
  seconds_ = elapsed(t0);
}

// =================================================================
// Testing aids

namespace {

/// (|set|, 4) components of sets[k], checking the bond.
template<class Sets>
torch::Tensor set_components(const Sets& sets, int64_t k, const char* where)
{
  if (k < 0 || k >= static_cast<int64_t>(sets.size()))
    throw utils::runtime_error(where, "No index set at that bond");
  const auto& set = sets[k];
  torch::Tensor out = torch::empty({set.size(), 4}, torch::kInt64);
  std::copy(set.comp.begin(), set.comp.end(), out.data_ptr<int64_t>());
  return out;
}

} // namespace

torch::Tensor LayoutCross::index_set(bool left, int64_t k) const
{
  return set_components(left ? left_ : right_, k, ctx("index_set").c_str());
}

const LayoutCross::PassRecord& LayoutCross::recorded(int64_t i) const
{
  if (i < 0 || i >= num_recorded())
    throw utils::runtime_error(ctx("recorded"), "No such recorded pass");
  return passes_[i];
}

bool LayoutCross::recorded_forward(int64_t i) const
{
  return recorded(i).forward;
}

std::vector<torch::Tensor> LayoutCross::recorded_cores(int64_t i) const
{
  return recorded(i).cores;
}

std::vector<torch::Tensor> LayoutCross::recorded_bases(int64_t i) const
{
  return recorded(i).bases;
}

torch::Tensor LayoutCross::recorded_index_set(
  int64_t i, bool left, int64_t k) const
{
  const auto& p = recorded(i);
  return set_components(
    left ? p.left : p.right, k, ctx("recorded_index_set").c_str());
}

std::vector<int64_t> LayoutCross::grown_block_keys(
  int64_t first, int64_t last) const
{
  if (first < 0 || last < first || last >= K_)
    throw utils::runtime_error(ctx("grown_block_keys"), "Invalid core range");
  auto it = grow_.find(core_mask(first, last));
  return it == grow_.end() ? std::vector<int64_t> {} : it->second;
}

std::tuple<torch::Tensor, std::vector<int64_t>> LayoutCross::candidate_grams(
  int64_t k, bool left, const torch::Tensor& core, bool factored) const
{
  if (k < 0 || k >= K_ || (left && k + 1 >= K_) || (!left && k < 1))
    throw utils::runtime_error(ctx("candidate_grams"), "No selection at core");
  const auto& side = left ? left_[k] : right_[k + 1];
  if (!side.defined)
    throw utils::runtime_error(ctx("candidate_grams"), "Run the cross first");
  const auto cb =
    candidate_blocks(k, left, core.to(torch::kFloat64).contiguous());
  return {block_grams(cb, factored), cb.keys};
}

torch::Tensor LayoutCross::core_components(int64_t k) const
{
  if (k < 0 || k >= K_)
    throw utils::runtime_error(ctx("core_components"), "No such core");
  auto out = torch::full({n_[k], 4}, -1, torch::kInt64);
  int64_t* c = out.data_ptr<int64_t>();
  for (int64_t j = 0; j < n_[k]; j++)
    decode(k, j, c + 4 * j);
  return out;
}

std::vector<int64_t> LayoutCross::forced_block_keys(
  int64_t first, int64_t last) const
{
  if (first < 0 || last < first || last >= K_)
    throw utils::runtime_error(ctx("forced_block_keys"), "Invalid core range");
  return forced_keys(core_mask(first, last));
}

} // namespace ttnte::cross
