#include "ttnte/cross/layout_cross.hpp"
#include "ttnte/cross/pivots.hpp"
#include "ttnte/linalg/banded_cholesky.hpp"
#include "ttnte/linalg/matrix_ops.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/utils/exception.hpp"
#include <ATen/Parallel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <set>

namespace ttnte::cross {

namespace {

using Clock = std::chrono::steady_clock;

/// Largest patch whose mass Cholesky factor is also kept dense.
constexpr int64_t DENSE_MASS_MAX = 3000;

double elapsed(Clock::time_point t0)
{
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

int mode_of(tensor::Dimension dim)
{
  switch (dim) {
  case tensor::Dimension::POLAR:
    return LayoutCross::MU;
  case tensor::Dimension::AZIMUTHAL:
    return LayoutCross::GA;
  case tensor::Dimension::ENERGY:
    return LayoutCross::E;
  default:
    return LayoutCross::S;
  }
}

/// Bit mask of the cores [first, last].
uint32_t core_mask(size_t first, size_t last)
{
  uint32_t mask = 0;
  for (size_t k = first; k <= last; k++)
    mask |= 1u << k;
  return mask;
}

/// Merge partial components: every mode is set by exactly one part.
inline void merge(int64_t* out, const int64_t* part)
{
  for (int m = 0; m < 4; m++)
    if (part[m] >= 0)
      out[m] = part[m];
}

/// Left inverse (B^T B)^{-1} B^T of a well-conditioned tall B, by Cholesky
/// of its Gram matrix (block selection keeps sigma_min / sigma_max >= 1e-2);
/// the pseudo-inverse if the Gram matrix is not safely definite.
torch::Tensor left_inverse(const torch::Tensor& B)
{
  auto [L, info] = torch::linalg_cholesky_ex(B.t().mm(B));
  if (info.item<int64_t>() == 0) {
    auto d = L.diagonal();
    if (d.min().item<double>() > 1e-3 * d.max().item<double>())
      return torch::cholesky_solve(B.t(), L);
  }
  return torch::linalg_pinv(B);
}

/// Rank-one factors of a nonnegative matrix that must be an outer product.
std::pair<torch::Tensor, torch::Tensor> outer_factors(
  const torch::Tensor& W, double tol)
{
  auto [U, s, Vh] = torch::linalg_svd(W, false);
  auto u = U.select(1, 0) * s[0];
  auto v = Vh.select(0, 0);
  if (v.sum().item<double>() < 0) {
    u = -u;
    v = -v;
  }
  const double residual = (W - u.outer(v)).norm().item<double>();
  if (residual > tol * std::max(W.norm().item<double>(), 1e-300))
    return {};
  return {u.contiguous(), v.contiguous()};
}

} // namespace

// =================================================================
// Construction and bookkeeping

int64_t LayoutCross::max_rank() const noexcept
{
  return config_.max_rank > 0 ? config_.max_rank
                              : std::numeric_limits<int64_t>::max();
}

LayoutCross::LayoutCross(CrossProblem problem, CrossConfig config)
  : problem_(std::move(problem)), config_(std::move(config)), rng_(config_.seed)
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
  if (config_.tol <= 0)
    throw utils::runtime_error(ctx("LayoutCross"), "tol must be positive");

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
    if (!config_.lethargy_widths.empty()) {
      if (static_cast<int64_t>(config_.lethargy_widths.size()) !=
          P.num_groups) {
        throw utils::runtime_error(
          ctx("LayoutCross"), "lethargy_widths must have one entry per group");
      }
      for (int64_t g = 0; g < P.num_groups; g++)
        energy_weight_[g] = 1.0 / std::sqrt(config_.lethargy_widths[g]);
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
  slot_.clear();
  fibers_.clear();
  swept_.assign(num_directions_ * size_[E], 0);
  num_swept_ = 0;
  force_.clear();
  grow_.clear();
  pool_.clear();
  sweep_seconds_ = 0.0;
  IndexSet unset;
  unset.width = -1;
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

void LayoutCross::ensure(std::vector<int64_t> pairs)
{
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                [&](int64_t p) { return swept_[p] != 0; }),
    pairs.end());
  if (pairs.empty())
    return;
  const int64_t G = size_[E];
  const int64_t Ns = size_[S];
  // Directions sharing a group list are swept as one block
  std::map<int64_t, std::vector<int64_t>> by_direction;
  for (int64_t p : pairs)
    by_direction[p / G].push_back(p % G);
  std::map<std::vector<int64_t>, std::vector<int64_t>> by_groups;
  for (auto& [d, groups] : by_direction)
    by_groups[groups].push_back(d);
  std::vector<sweep::FiberBlock> blocks;
  for (auto& [groups, dirs] : by_groups)
    blocks.push_back({dirs, groups});

  const int64_t first = static_cast<int64_t>(slot_.size());
  const int64_t count = static_cast<int64_t>(pairs.size());
  fibers_.resize((first + count) * Ns);
  auto t0 = Clock::now();
  problem_.sweep(blocks, fibers_.data() + first * Ns);
  sweep_seconds_ += elapsed(t0);
  timings_["sweep"] += elapsed(t0);
  t0 = Clock::now();

  std::vector<int64_t> new_pairs;
  new_pairs.reserve(count);
  for (const auto& blk : blocks)
    for (int64_t d : blk.directions)
      for (int64_t g : blk.groups)
        new_pairs.push_back(d * G + g);
  for (int64_t i = 0; i < count; i++) {
    slot_[new_pairs[i]] = first + i;
    swept_[new_pairs[i]] = 1;
  }
  num_swept_ += count;

  // Weight the new samples: R_P per patch (BLAS-3 on the batch where the
  // dense factor is held), then sqrt(w_d) / sqrt(du_g)
  auto F =
    torch::from_blob(fibers_.data() + first * Ns, {count, Ns}, torch::kFloat64);
  for (size_t q = 0; q < mass_factors_.size(); q++) {
    const int64_t off = problem_.patch_offsets[q];
    const int64_t n = mass_factors_[q].size(0);
    if (mass_dense_[q].defined()) {
      auto block = F.narrow(1, off, n);
      block.copy_(block.mm(mass_dense_[q].t()));
      continue;
    }
    at::parallel_for(0, count, 1, [&](int64_t b0, int64_t b1) {
      for (int64_t i = b0; i < b1; i++)
        linalg::banded::cholesky_apply_upper(
          mass_factors_[q].data_ptr<double>(), n, mass_kd_[q],
          fibers_.data() + (first + i) * Ns + off);
    });
  }
  std::vector<double> scale(count);
  for (int64_t i = 0; i < count; i++)
    scale[i] =
      angle_weight_[new_pairs[i] / G] * energy_weight_[new_pairs[i] % G];
  F.mul_(torch::from_blob(scale.data(), {count, 1}, torch::kFloat64));
  timings_["weight"] += elapsed(t0);
}

// =================================================================
// Index sets

LayoutCross::IndexSet LayoutCross::candidates_left(
  size_t k, const std::vector<int64_t>& rows) const
{
  const auto& L = left_[k];
  IndexSet out;
  out.width = static_cast<int64_t>(k) + 1;
  out.idx.reserve(rows.size() * out.width);
  out.comp.reserve(rows.size() * 4);
  for (int64_t c : rows) {
    const int64_t i = c / n_[k], j = c % n_[k];
    for (int64_t w = 0; w < L.width; w++)
      out.idx.push_back(L.idx[i * L.width + w]);
    out.idx.push_back(j);
    int64_t comp[4];
    std::copy(&L.comp[i * 4], &L.comp[i * 4] + 4, comp);
    decode(k, j, comp);
    out.comp.insert(out.comp.end(), comp, comp + 4);
  }
  return out;
}

LayoutCross::IndexSet LayoutCross::candidates_right(
  size_t k, const std::vector<int64_t>& rows) const
{
  const auto& R = right_[k + 1];
  const int64_t mr = R.size();
  IndexSet out;
  out.width = K_ - static_cast<int64_t>(k);
  out.idx.reserve(rows.size() * out.width);
  out.comp.reserve(rows.size() * 4);
  for (int64_t c : rows) {
    const int64_t j = c / mr, l = c % mr;
    out.idx.push_back(j);
    for (int64_t w = 0; w < R.width; w++)
      out.idx.push_back(R.idx[l * R.width + w]);
    int64_t comp[4];
    std::copy(&R.comp[l * 4], &R.comp[l * 4] + 4, comp);
    decode(k, j, comp);
    out.comp.insert(out.comp.end(), comp, comp + 4);
  }
  return out;
}

std::vector<int64_t> LayoutCross::candidate_keys_right(size_t k) const
{
  const auto& R = right_[k + 1];
  const int64_t mr = R.size(), nk = n_[k];
  std::vector<int64_t> keys(nk * mr);
  at::parallel_for(0, nk, 1, [&](int64_t j0, int64_t j1) {
    for (int64_t j = j0; j < j1; j++)
      for (int64_t l = 0; l < mr; l++) {
        int64_t comp[4];
        std::copy(&R.comp[l * 4], &R.comp[l * 4] + 4, comp);
        decode(k, j, comp);
        keys[j * mr + l] = row_key(comp);
      }
  });
  return keys;
}

std::vector<int64_t> LayoutCross::set_keys(const IndexSet& set) const
{
  std::vector<int64_t> keys(set.size());
  for (int64_t i = 0; i < set.size(); i++)
    keys[i] = row_key(&set.comp[i * 4]);
  return keys;
}

// =================================================================
// Superblocks

torch::Tensor LayoutCross::superblock(size_t k, bool raw)
{
  const auto& L = left_[k];
  const auto& R = right_[k + 1];
  const int64_t ml = L.size(), nk = n_[k], mr = R.size();
  const int64_t Ns = size_[S];

  // Which part enumerates space: the left rows, the core, or the right rows
  const auto& modes = core_modes_[k];
  const auto qs_it = std::find(modes.begin(), modes.end(), static_cast<int>(S));
  const int holder = L.comp[S] >= 0 ? 0 : (qs_it != modes.end() ? 1 : 2);

  // Group the holder's rows by their non-space components
  struct Group {
    std::array<int64_t, 4> comp;
    std::vector<std::pair<int64_t, int64_t>> rows; // (row, s)
  };
  std::vector<Group> groups;
  if (holder != 1) {
    const auto& H = holder == 0 ? L : R;
    std::unordered_map<int64_t, size_t> where;
    for (int64_t i = 0; i < H.size(); i++) {
      const int64_t* c = &H.comp[i * 4];
      auto [it, fresh] = where.emplace(row_key(c), groups.size());
      if (fresh) {
        Group g;
        std::copy(c, c + 4, g.comp.begin());
        g.comp[S] = -1;
        groups.push_back(std::move(g));
      }
      groups[it->second].rows.emplace_back(i, c[S]);
    }
  }
  // Core with space: j = (o * Ns + s) * inner + ci
  int64_t inner = 1, outer = 1;
  if (holder == 1) {
    const size_t qs = qs_it - modes.begin();
    for (size_t q = qs + 1; q < modes.size(); q++)
      inner *= core_dims_[k][q];
    outer = nk / (Ns * inner);
  }

  // Visit every (pair, destination) run; `emit(pair, slots...)` style via
  // a callback taking the pair and a writer
  auto visit = [&](auto&& on_pair, int64_t a0, int64_t a1) {
    int64_t comp[4];
    if (holder == 1) {
      for (int64_t i = a0; i < a1; i++)
        for (int64_t o = 0; o < outer; o++)
          for (int64_t ci = 0; ci < inner; ci++)
            for (int64_t l = 0; l < mr; l++) {
              std::copy(&L.comp[i * 4], &L.comp[i * 4] + 4, comp);
              decode(k, o * Ns * inner + ci, comp);
              merge(comp, &R.comp[l * 4]);
              const int64_t base = (i * nk + o * Ns * inner + ci) * mr + l;
              on_pair(pair_of(comp), [&](const double* f, double* out) {
                for (int64_t s = 0; s < Ns; s++)
                  out[base + s * inner * mr] = f[s];
              });
            }
    } else if (holder == 0) {
      for (int64_t gi = a0; gi < a1; gi++)
        for (int64_t j = 0; j < nk; j++)
          for (int64_t l = 0; l < mr; l++) {
            std::copy(groups[gi].comp.begin(), groups[gi].comp.end(), comp);
            int64_t cj[4] = {-1, -1, -1, -1};
            decode(k, j, cj);
            merge(comp, cj);
            merge(comp, &R.comp[l * 4]);
            on_pair(pair_of(comp), [&](const double* f, double* out) {
              for (const auto& [i, s] : groups[gi].rows)
                out[(i * nk + j) * mr + l] = f[s];
            });
          }
    } else {
      for (int64_t i = a0; i < a1; i++)
        for (int64_t j = 0; j < nk; j++)
          for (size_t gi = 0; gi < groups.size(); gi++) {
            std::copy(&L.comp[i * 4], &L.comp[i * 4] + 4, comp);
            int64_t cj[4] = {-1, -1, -1, -1};
            decode(k, j, cj);
            merge(comp, cj);
            merge(comp, groups[gi].comp.data());
            on_pair(pair_of(comp), [&](const double* f, double* out) {
              for (const auto& [l, s] : groups[gi].rows)
                out[(i * nk + j) * mr + l] = f[s];
            });
          }
    }
  };
  const int64_t range = holder == 0 ? static_cast<int64_t>(groups.size()) : ml;

  // Pass 1: the pairs needed
  std::vector<int64_t> pairs;
  visit(
    [&](int64_t pair, auto&&) {
      if (!swept_[pair])
        pairs.push_back(pair);
    },
    0, range);
  ensure(std::move(pairs));

  // Pass 2: copy the samples
  auto out_t = torch::empty({ml, nk, mr}, torch::kFloat64);
  double* out = out_t.data_ptr<double>();
  at::parallel_for(0, range, 1, [&](int64_t a0, int64_t a1) {
    int64_t last = -1;
    const double* f = nullptr;
    visit(
      [&](int64_t pair, auto&& write) {
        if (pair != last) {
          f = fiber(pair);
          last = pair;
        }
        write(f, out);
      },
      a0, a1);
  });
  if (raw)
    return out_t;
  if (PL_[k].defined())
    out_t = torch::tensordot(PL_[k], out_t, {1}, {0});
  if (PR_[k + 1].defined())
    out_t = torch::tensordot(out_t, PR_[k + 1], {2}, {0});
  return out_t;
}

// =================================================================
// Block selection

double LayoutCross::block_cost(int64_t key) const
{
  const int64_t G = size_[E], Ng = size_[GA];
  const int64_t e = key % (G + 1) - 1;
  const int64_t rest = key / (G + 1);
  const int64_t ga = rest % (Ng + 1) - 1;
  const int64_t mu = rest / (Ng + 1) - 1;
  const int64_t m0 = mu >= 0 ? mu : 0, m1 = mu >= 0 ? mu + 1 : size_[MU];
  const int64_t a0 = ga >= 0 ? ga : 0, a1 = ga >= 0 ? ga + 1 : Ng;
  const int64_t e0 = e >= 0 ? e : 0, e1 = e >= 0 ? e + 1 : G;
  double unswept = 0.0;
  for (int64_t m = m0; m < m1; m++)
    for (int64_t a = a0; a < a1; a++) {
      const uint8_t* row = &swept_[(m * Ng + a) * G];
      for (int64_t g = e0; g < e1; g++)
        unswept += row[g] ? 0.0 : 1.0;
    }
  return unswept;
}

std::vector<int64_t> LayoutCross::forced_keys(uint32_t mask) const
{
  bool present[4] = {false, false, false, false};
  for (int64_t k = 0; k < K_; k++)
    if (mask & (1u << k))
      for (int m : core_modes_[k])
        present[m] = true;
  const int64_t G = size_[E], Ng = size_[GA];
  std::vector<int64_t> keys;
  for (int64_t p : force_) {
    const int64_t d = p / G;
    keys.push_back(key(present[MU] ? d / Ng : -1, present[GA] ? d % Ng : -1,
      present[E] ? p % G : -1));
  }
  return keys;
}

std::pair<std::vector<int64_t>, torch::Tensor> LayoutCross::select_candidates(
  size_t k, bool left, const torch::Tensor& core,
  const std::vector<int64_t>& keep)
{
  const IndexSet& side = left ? left_[k] : right_[k + 1];
  const int64_t ms = side.size();
  const int64_t nk = n_[k];
  // Candidate row (a, j) of the new basis is S_a K_j: S the side vectors
  // (rows of B_k, or columns of C_{k+1}), K_j a slice of the core
  torch::Tensor Sv, K;
  if (left) {
    Sv = B_[k].defined() ? B_[k] : torch::eye(ms, torch::kFloat64);
    K = core.permute({1, 0, 2});
  } else {
    Sv = C_[k + 1].defined() ? C_[k + 1].t() : torch::eye(ms, torch::kFloat64);
    K = core.permute({1, 2, 0});
  }
  Sv = Sv.contiguous();
  K = K.contiguous();
  const int64_t q = Sv.size(1), r = K.size(2);

  // Side rows and core values grouped by their (non-space) key parts
  struct Group {
    std::array<int64_t, 4> comp;
    std::vector<int64_t> members;
  };
  auto group = [&](int64_t count, auto&& comp_of) {
    std::vector<Group> groups;
    std::unordered_map<int64_t, size_t> where;
    for (int64_t x = 0; x < count; x++) {
      std::array<int64_t, 4> c;
      comp_of(x, c.data());
      c[S] = -1;
      auto [it, fresh] = where.emplace(row_key(c.data()), groups.size());
      if (fresh)
        groups.push_back({c, {}});
      groups[it->second].members.push_back(x);
    }
    return groups;
  };
  auto side_groups = group(ms, [&](int64_t a, int64_t* c) {
    std::copy(&side.comp[a * 4], &side.comp[a * 4] + 4, c);
  });
  auto core_groups = group(nk, [&](int64_t j, int64_t* c) {
    std::fill(c, c + 4, -1);
    decode(k, j, c);
  });

  // One block per (side group, core group). Single-row side groups (the
  // common case) use the candidate rows T[a, j] = S_a K_j directly, batched;
  // multi-row groups use G = sum_j K_j^T H_A K_j, H_A = S_A^T S_A
  const int64_t nA = static_cast<int64_t>(side_groups.size());
  const int64_t nJ = static_cast<int64_t>(core_groups.size());
  const int64_t nb = nA * nJ;
  bool single = true;
  for (const auto& g : side_groups)
    single = single && g.members.size() == 1;
  std::vector<torch::Tensor> J_index(nJ);
  for (int64_t J = 0; J < nJ; J++)
    J_index[J] = torch::tensor(core_groups[J].members, torch::kInt64);
  auto A_order = torch::empty({ms}, torch::kInt64);
  {
    int64_t pos = 0;
    auto* o = A_order.data_ptr<int64_t>();
    for (const auto& g : side_groups)
      for (int64_t x : g.members)
        o[pos++] = x;
  }
  // Rows of the side reordered by group: T[a, j] for a in group order
  torch::Tensor T;
  if (single)
    T = torch::einsum("aq,jqr->ajr", {Sv.index_select(0, A_order), K});
  bool single_rows = single;
  for (const auto& g : core_groups)
    single_rows = single_rows && g.members.size() == 1;
  // Block Grams, unless every block is one row (selected on the rows)
  torch::Tensor grams;
  if (!single_rows)
    grams = torch::empty({nA, nJ, r, r}, torch::kFloat64);
  if (single_rows) {
  } else if (single) {
    for (int64_t J = 0; J < nJ; J++) {
      auto TJ = T.index_select(1, J_index[J]); // (nA, |J|, r)
      grams.select(1, J).copy_(torch::bmm(TJ.transpose(1, 2), TJ));
    }
  } else {
    auto H = torch::empty({nA, q, q}, torch::kFloat64);
    for (int64_t A = 0; A < nA; A++) {
      auto SA = Sv.index_select(
        0, torch::tensor(side_groups[A].members, torch::kInt64));
      H[A].copy_(SA.t().mm(SA));
    }
    for (int64_t J = 0; J < nJ; J++) {
      auto KJ = K.index_select(0, J_index[J]); // (|J|, q, r)
      grams.select(1, J).copy_(torch::einsum("jqr,aqp,jps->ars", {KJ, H, KJ}));
    }
  }
  if (!single_rows)
    grams = grams.reshape({nb, r, r});
  std::vector<int64_t> sizes(nb), keys(nb);
  std::vector<double> cost(nb);
  for (int64_t bk = 0; bk < nb; bk++) {
    const int64_t A = bk / nJ, J = bk % nJ;
    int64_t c[4];
    std::copy(side_groups[A].comp.begin(), side_groups[A].comp.end(), c);
    merge(c, core_groups[J].comp.data());
    keys[bk] = row_key(c);
    sizes[bk] = static_cast<int64_t>(
      side_groups[A].members.size() * core_groups[J].members.size());
    cost[bk] = block_cost(keys[bk]);
  }
  // Forced pairs, the current set and residual-grown blocks are always kept
  const uint32_t mask = left ? core_mask(0, k) : core_mask(k, K_ - 1);
  std::vector<int64_t> start = forced_keys(mask);
  start.insert(start.end(), keep.begin(), keep.end());
  auto it = grow_.find(mask);
  if (it != grow_.end())
    start.insert(start.end(), it->second.begin(), it->second.end());
  std::sort(start.begin(), start.end());
  std::vector<uint8_t> picked(nb, 0);
  for (int64_t b = 0; b < nb; b++)
    picked[b] = std::binary_search(start.begin(), start.end(), keys[b]);
  if (single_rows) {
    // One row per block: greedy Gram-Schmidt on the rows themselves
    std::vector<int64_t> core_index(nJ);
    for (int64_t J = 0; J < nJ; J++)
      core_index[J] = core_groups[J].members[0];
    auto rows_t = T.index_select(1, torch::tensor(core_index, torch::kInt64))
                    .reshape({-1, r});
    select_single_rows(rows_t, cost, picked);
  } else {
    select_block_grams(grams, sizes, cost, picked);
  }

  // The chosen rows, ascending, and the new basis there
  std::vector<int64_t> rows;
  std::vector<torch::Tensor> parts;
  std::vector<int64_t> group_start(nA + 1, 0);
  for (int64_t A = 0; A < nA; A++)
    group_start[A + 1] =
      group_start[A] + static_cast<int64_t>(side_groups[A].members.size());
  for (int64_t bk = 0; bk < nb; bk++) {
    if (!picked[bk])
      continue;
    const int64_t A = bk / nJ, J = bk % nJ;
    for (int64_t a : side_groups[A].members)
      for (int64_t j : core_groups[J].members)
        rows.push_back(left ? a * nk + j : j * ms + a);
    torch::Tensor part;
    if (single) {
      part = T.narrow(0, group_start[A], 1).index_select(1, J_index[J]);
    } else {
      auto SA = Sv.index_select(
        0, torch::tensor(side_groups[A].members, torch::kInt64));
      part = torch::einsum("aq,jqr->ajr", {SA, K.index_select(0, J_index[J])});
    }
    parts.push_back(part.reshape({-1, r}));
  }
  auto basis = torch::cat(parts, 0);
  std::vector<int64_t> order(rows.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
    [&](int64_t x, int64_t y) { return rows[x] < rows[y]; });
  std::vector<int64_t> sorted(rows.size());
  for (size_t i = 0; i < order.size(); i++)
    sorted[i] = rows[order[i]];
  basis = basis.index_select(0, torch::tensor(order, torch::kInt64));
  return {sorted, basis};
}

void LayoutCross::grow_from_residual(const torch::Tensor& col, double total,
  size_t k, uint32_t mask, const IndexSet& current)
{
  auto c = col.contiguous();
  const double* p = c.data_ptr<double>();
  const int64_t n = c.numel();
  double sum = 0.0;
  for (int64_t v = 0; v < n; v++)
    sum += p[v];
  if (std::sqrt(sum) <= config_.tol * total)
    return;
  std::vector<int64_t> vals;
  for (int64_t v = 0; v < n; v++)
    if (p[v] > 0)
      vals.push_back(v);
  std::sort(vals.begin(), vals.end(),
    [p](int64_t a, int64_t b) { return p[a] > p[b]; });
  int64_t need = 0;
  for (double acc = 0.0; need < static_cast<int64_t>(vals.size());) {
    acc += p[vals[need++]];
    if (acc >= 0.5 * sum)
      break;
  }
  auto current_keys = set_keys(current);
  std::sort(current_keys.begin(), current_keys.end());
  const int64_t nblocks =
    std::unique(current_keys.begin(), current_keys.end()) -
    current_keys.begin();
  const int64_t take = std::min(need, std::max<int64_t>(4, nblocks / 2));
  auto& grown = grow_[mask];
  for (int64_t i = 0; i < take; i++) {
    int64_t comp[4] = {-1, -1, -1, -1};
    decode(k, vals[i], comp);
    const int64_t key = row_key(comp);
    if (std::find(grown.begin(), grown.end(), key) == grown.end())
      grown.push_back(key);
  }
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
  if (D0.empty())
    D0.push_back(0);
  std::vector<int64_t> pairs;
  for (int64_t d : D0)
    for (int64_t g = 0; g < G; g++)
      pairs.push_back(d * G + g);
  ensure(pairs);

  // Initial energies: pivots of a Gaussian sketch of the swept rows
  std::vector<int64_t> J0;
  for (int64_t kk = 32;; kk *= 2) {
    auto sketch = torch::zeros({kk, G}, torch::kFloat64);
    for (int64_t d : D0) {
      auto F = torch::empty({G, Ns}, torch::kFloat64);
      for (int64_t g = 0; g < G; g++)
        std::copy(fiber(d * G + g), fiber(d * G + g) + Ns,
          F.data_ptr<double>() + g * Ns);
      sketch += direction_sketch(d, kk, Ns).mm(F.t());
    }
    J0 = pivot_columns(sketch, config_.tol);
    if (static_cast<int64_t>(J0.size()) <= kk / 2 ||
        kk >= std::min<int64_t>(G, static_cast<int64_t>(D0.size()) * Ns))
      break;
  }
  std::set<int64_t> J0s(J0.begin(), J0.end()), D0s(D0.begin(), D0.end());
  std::set<int64_t> mu0, ga0;
  for (int64_t d : D0) {
    mu0.insert(d / Ng);
    ga0.insert(d % Ng);
  }

  for (int64_t k = K_ - 1; k >= 1; k--) {
    const auto& R = right_[k + 1];
    const int64_t mr = R.size();
    bool has[4] = {false, false, false, false};
    for (int64_t c = k; c < K_; c++)
      for (int m : core_modes_[c])
        has[m] = true;
    std::vector<int64_t> rows;
    for (int64_t j = 0; j < n_[k]; j++)
      for (int64_t l = 0; l < mr; l++) {
        int64_t comp[4];
        std::copy(&R.comp[l * 4], &R.comp[l * 4] + 4, comp);
        decode(k, j, comp);
        bool ok = true;
        if (has[E])
          ok = ok && J0s.count(comp[E]);
        if (has[MU] && has[GA])
          ok = ok && D0s.count(comp[MU] * Ng + comp[GA]);
        else if (has[MU])
          ok = ok && mu0.count(comp[MU]);
        else if (has[GA])
          ok = ok && ga0.count(comp[GA]);
        if (ok)
          rows.push_back(j * mr + l);
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
    right_[k] = candidates_right(k, rows);
    C_[k] = torch::Tensor();
    PR_[k] = torch::Tensor();
  }
}

void LayoutCross::forward()
{
  // Right sides as interpolation on their samples: ranks may grow up to them
  for (int64_t k = 0; k + 1 < K_; k++) {
    C_[k + 1] = torch::Tensor();
    PR_[k + 1] = torch::Tensor();
  }
  const double share = 0.1 / std::sqrt(std::max<double>(K_ - 1, 1));
  for (int64_t k = 0; k + 1 < K_; k++) {
    auto t0 = Clock::now();
    auto Sb = superblock(k, false);
    timings_["superblock"] += elapsed(t0);
    t0 = Clock::now();
    const int64_t rl = Sb.size(0), nk = Sb.size(1), rr = Sb.size(2);
    auto [U, s, Vh, delta, rest] = linalg::truncated_svd(
      Sb.reshape({rl * nk, rr}), share * config_.tol, max_rank(), true);
    const int64_t r = U.size(1);
    auto core = U.reshape({rl, nk, r}).contiguous();
    cores_[k] = core;
    timings_["svd"] += elapsed(t0);
    t0 = Clock::now();
    // Index sets only grow within a transport solve
    std::vector<int64_t> keep;
    if (left_[k + 1].width == k + 1)
      keep = set_keys(left_[k + 1]);
    auto [rows, basis] = select_candidates(k, true, core, keep);
    left_[k + 1] = candidates_left(k, rows);
    B_[k + 1] = basis;
    PL_[k + 1] = left_inverse(B_[k + 1]);
    timings_["select"] += elapsed(t0);
  }
  // Last core, fitted on its superblock; its residual grows the bond
  auto t0 = Clock::now();
  auto Sraw = superblock(K_ - 1, true);
  timings_["superblock"] += elapsed(t0);
  t0 = Clock::now();
  auto core = PL_[K_ - 1].defined()
                ? torch::tensordot(PL_[K_ - 1], Sraw, {1}, {0})
                : Sraw;
  cores_[K_ - 1] = core.contiguous();
  if (K_ > 1) {
    auto fit = B_[K_ - 1].defined()
                 ? torch::tensordot(B_[K_ - 1], core, {1}, {0})
                 : core;
    auto col = (Sraw - fit).select(2, 0).square().sum(0);
    grow_from_residual(col, Sraw.norm().item<double>(), K_ - 1,
      core_mask(K_ - 1, K_ - 1), right_[K_ - 1]);
  }
  timings_["svd"] += elapsed(t0);
}

void LayoutCross::backward()
{
  const double share = 1.0 / std::sqrt(std::max<double>(K_ - 1, 1));
  for (int64_t k = K_ - 1; k >= 1; k--) {
    auto t0 = Clock::now();
    auto Sb = superblock(k, false);
    timings_["superblock"] += elapsed(t0);
    t0 = Clock::now();
    const int64_t rl = Sb.size(0), nk = Sb.size(1), rr = Sb.size(2);
    auto [U, s, Vh, delta, rest] = linalg::truncated_svd(
      Sb.reshape({rl, nk * rr}), share * config_.tol, max_rank(), true);
    const int64_t r = Vh.size(0);
    auto core = Vh.reshape({r, nk, rr}).contiguous();
    cores_[k] = core;
    timings_["svd"] += elapsed(t0);
    t0 = Clock::now();
    auto [rows, basis] = select_candidates(k, false, core, set_keys(right_[k]));
    right_[k] = candidates_right(k, rows);
    C_[k] = basis.t();
    PR_[k] = left_inverse(basis).t();
    timings_["select"] += elapsed(t0);
  }
  auto t0 = Clock::now();
  auto Sraw = superblock(0, true);
  timings_["superblock"] += elapsed(t0);
  cores_[0] =
    (PR_[1].defined() ? torch::tensordot(Sraw, PR_[1], {2}, {0}) : Sraw)
      .contiguous();
}

double LayoutCross::check()
{
  const int64_t total = num_directions_ * size_[E];
  const int64_t holdout = std::max<int64_t>(config_.holdout, 1);
  const int64_t free = total - num_swept_;
  if (free >= holdout) {
    std::vector<int64_t> pick;
    if (free > total / 4) {
      std::uniform_int_distribution<int64_t> dist(0, total - 1);
      std::set<int64_t> chosen;
      while (static_cast<int64_t>(chosen.size()) < holdout) {
        const int64_t p = dist(rng_);
        if (!swept_[p])
          chosen.insert(p);
      }
      pick.assign(chosen.begin(), chosen.end());
    } else {
      std::vector<int64_t> all;
      for (int64_t p = 0; p < total; p++)
        if (!swept_[p])
          all.push_back(p);
      std::sample(
        all.begin(), all.end(), std::back_inserter(pick), holdout, rng_);
    }
    ensure(pick);
    pool_.insert(pool_.end(), pick.begin(), pick.end());
  } else {
    // (Nearly) everything swept: old hold-out fibers may have become pivots,
    // so judge on fresh random swept fibers
    std::vector<int64_t> all;
    for (int64_t p = 0; p < total; p++)
      if (swept_[p])
        all.push_back(p);
    pool_.clear();
    std::sample(
      all.begin(), all.end(), std::back_inserter(pool_), 4 * holdout, rng_);
  }
  if (pool_.empty())
    return 0.0;

  const int64_t B = static_cast<int64_t>(pool_.size());
  const int64_t G = size_[E], Ng = size_[GA], Ns = size_[S];
  std::vector<int64_t> mu(B), ga(B), e(B);
  for (int64_t b = 0; b < B; b++) {
    mu[b] = pool_[b] / G / Ng;
    ga[b] = pool_[b] / G % Ng;
    e[b] = pool_[b] % G;
  }
  std::array<Selector, 4> sel;
  sel[MU] = {Selector::Kind::BATCH, torch::tensor(mu, torch::kInt64)};
  sel[GA] = {Selector::Kind::BATCH, torch::tensor(ga, torch::kInt64)};
  sel[E] = {Selector::Kind::BATCH, torch::tensor(e, torch::kInt64)};
  auto approx = contract(sel).select(2, 0);
  auto exact = torch::empty({B, Ns}, torch::kFloat64);
  for (int64_t b = 0; b < B; b++)
    std::copy(
      fiber(pool_[b]), fiber(pool_[b]) + Ns, exact.data_ptr<double>() + b * Ns);
  auto err_b = (approx - exact).square().sum(1);
  const double err = std::sqrt(err_b.sum().item<double>()) /
                     std::max(exact.norm().item<double>(), 1e-300);
  if (err > config_.holdout_factor * config_.tol) {
    auto order = std::get<1>(err_b.sort(0, true));
    for (int64_t i = 0; i < std::min<int64_t>(4, B); i++) {
      const int64_t p = pool_[order[i].item<int64_t>()];
      if (std::find(force_.begin(), force_.end(), p) == force_.end())
        force_.push_back(p);
    }
  }
  return err;
}

void LayoutCross::unweight()
{
  const bool angle = config_.weighted;
  const bool energy = config_.weighted && !config_.lethargy_widths.empty();
  const bool space = !mass_factors_.empty();
  if (!angle && !energy && !space)
    return;
  auto w = problem_.weights.to(torch::kFloat64);
  for (int64_t k = 0; k < K_; k++) {
    const auto& modes = core_modes_[k];
    std::vector<int64_t> shape {cores_[k].size(0)};
    shape.insert(shape.end(), core_dims_[k].begin(), core_dims_[k].end());
    shape.push_back(cores_[k].size(2));
    auto t = cores_[k].reshape(shape).clone();
    const int64_t q = static_cast<int64_t>(modes.size());
    auto axis_of = [&](int mode) -> int64_t {
      for (int64_t a = 0; a < q; a++)
        if (modes[a] == mode)
          return a + 1;
      return -1;
    };
    auto scale_axis = [&](int64_t axis, const torch::Tensor& v) {
      std::vector<int64_t> view(t.dim(), 1);
      view[axis] = v.numel();
      t = t * v.reshape(view);
    };
    const int64_t amu = axis_of(MU), aga = axis_of(GA);
    if (angle) {
      if (amu >= 0 && aga >= 0) {
        std::vector<int64_t> view(t.dim(), 1);
        view[amu] = size_[MU];
        view[aga] = size_[GA];
        auto inv = w.reshape({size_[MU], size_[GA]}).rsqrt();
        if (amu > aga)
          inv = inv.t();
        t = t * inv.reshape(view);
      } else if (amu >= 0 && aga < 0 && size_[GA] == 1) {
        scale_axis(amu, w.rsqrt());
      } else {
        if (amu >= 0)
          scale_axis(amu, polar_weight_.rsqrt());
        if (aga >= 0)
          scale_axis(aga, azimuthal_weight_.rsqrt());
      }
    }
    const int64_t ae = axis_of(E);
    if (energy && ae >= 0) {
      auto ew = torch::tensor(energy_weight_, torch::kFloat64);
      scale_axis(ae, ew.reciprocal());
    }
    const int64_t as = axis_of(S);
    if (space && as >= 0) {
      auto moved = t.movedim(as, -1).contiguous();
      auto flat = moved.reshape({-1, size_[S]});
      double* data = flat.data_ptr<double>();
      const int64_t rows = flat.size(0);
      for (size_t p = 0; p < mass_factors_.size(); p++) {
        const int64_t off = problem_.patch_offsets[p];
        const int64_t n = mass_factors_[p].size(0);
        if (mass_dense_[p].defined()) {
          auto block = flat.narrow(1, off, n);
          block.copy_(torch::linalg_solve_triangular(
            mass_dense_[p], block.t(), /*upper=*/true)
              .t());
          continue;
        }
        at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
          for (int64_t r = r0; r < r1; r++)
            linalg::banded::cholesky_solve_upper(
              mass_factors_[p].data_ptr<double>(), n, mass_kd_[p],
              data + r * size_[S] + off);
        });
      }
      t = moved.movedim(-1, as);
    }
    cores_[k] =
      t.reshape({cores_[k].size(0), n_[k], cores_[k].size(2)}).contiguous();
  }
}

void LayoutCross::run()
{
  auto t0 = Clock::now();
  reset();
  history_.clear();
  auto t1 = Clock::now();
  init_right();
  timings_["init"] += elapsed(t1);
  const double accept = config_.holdout_factor * config_.tol;
  double err = std::numeric_limits<double>::infinity();
  sweeps_done_ = 0;
  for (int64_t it = 0; it < std::max<int64_t>(config_.max_sweeps, 1); it++) {
    forward();
    history_.push_back(
      {true, ranks(), num_swept_, std::numeric_limits<double>::quiet_NaN()});
    backward();
    sweeps_done_ = it + 1;
    t1 = Clock::now();
    err = check();
    timings_["check"] += elapsed(t1);
    history_.push_back({false, ranks(), num_swept_, err});
    if (err <= accept)
      break;
  }
  holdout_error_ = err;
  converged_ = err <= accept;
  t1 = Clock::now();
  unweight();
  timings_["unweight"] += elapsed(t1);
  total_fibers_ += num_swept_;
  fibers_.clear();
  fibers_.shrink_to_fit();
  slot_.clear();
  seconds_ = elapsed(t0);
}

// =================================================================
// Contractions

torch::Tensor LayoutCross::contract(const std::array<Selector, 4>& select,
  const torch::Tensor& angle_matrix) const
{
  using Kind = Selector::Kind;
  if (!has_tt())
    throw utils::runtime_error(ctx("contract"),
      "No TT held: run() first, or set keep_tt to keep it after a solve");
  const char letter[4] = {'m', 'g', 's', 'e'};
  int64_t core_of_mode[4] = {-1, -1, -1, -1};
  for (int64_t k = 0; k < K_; k++)
    for (int m : core_modes_[k])
      core_of_mode[m] = k;
  const bool joint = select[MU].kind == Kind::WEIGHT &&
                     select[GA].kind == Kind::WEIGHT && core_of_mode[GA] >= 0 &&
                     core_of_mode[MU] == core_of_mode[GA];

  torch::Tensor env;
  bool env_batch = false;
  std::string env_free;
  for (int64_t k = 0; k < K_; k++) {
    std::vector<int64_t> shape {cores_[k].size(0)};
    shape.insert(shape.end(), core_dims_[k].begin(), core_dims_[k].end());
    shape.push_back(cores_[k].size(2));
    auto t = cores_[k].reshape(shape);
    std::vector<int> axes(core_modes_[k].begin(), core_modes_[k].end());
    auto axis_of = [&](int mode) {
      return static_cast<int64_t>(
               std::find(axes.begin(), axes.end(), mode) - axes.begin()) +
             1;
    };
    for (int m : core_modes_[k])
      if (select[m].kind == Kind::LIST)
        t = t.index_select(axis_of(m), select[m].data);
    if (joint && core_of_mode[MU] == k) {
      t = torch::tensordot(t, angle_matrix, {axis_of(MU), axis_of(GA)}, {0, 1});
      // tensordot puts the (empty) result dims last: r_r stays last
      axes.erase(std::find(axes.begin(), axes.end(), MU));
      axes.erase(std::find(axes.begin(), axes.end(), GA));
    }
    for (int m : core_modes_[k]) {
      if (select[m].kind != Kind::WEIGHT || (joint && (m == MU || m == GA)))
        continue;
      t = torch::tensordot(t, select[m].data, {axis_of(m)}, {0});
      axes.erase(std::find(axes.begin(), axes.end(), m));
    }
    // Batched modes: gather their joint index into a leading batch axis
    std::vector<int> batch;
    for (int m : axes)
      if (select[m].kind == Kind::BATCH)
        batch.push_back(m);
    const bool core_batch = !batch.empty();
    std::string free;
    if (core_batch) {
      std::vector<int64_t> perm;
      for (int m : batch)
        perm.push_back(axis_of(m));
      perm.push_back(0);
      for (size_t a = 0; a < axes.size(); a++)
        if (select[axes[a]].kind != Kind::BATCH)
          perm.push_back(static_cast<int64_t>(a) + 1);
      perm.push_back(t.dim() - 1);
      t = t.permute(perm).contiguous();
      int64_t prod = 1;
      torch::Tensor lin;
      for (int m : batch) {
        lin = lin.defined() ? lin * size_[m] + select[m].data
                            : select[m].data.clone();
        prod *= size_[m];
      }
      std::vector<int64_t> flat {prod};
      for (int64_t a = static_cast<int64_t>(batch.size()); a < t.dim(); a++)
        flat.push_back(t.size(a));
      t = t.reshape(flat).index_select(0, lin);
      for (int m : axes)
        if (select[m].kind != Kind::BATCH)
          free += letter[m];
    } else {
      for (int m : axes)
        free += letter[m];
    }
    if (!env.defined()) {
      env = t.select(core_batch ? 1 : 0, 0);
      env_batch = core_batch;
      env_free = free;
      continue;
    }
    const bool out_batch = env_batch || core_batch;
    const std::string spec = std::string(env_batch ? "b" : "") + env_free +
                             "q," + (core_batch ? "b" : "") + "q" + free +
                             "w->" + (out_batch ? "b" : "") + env_free + free +
                             "w";
    env = torch::einsum(spec, {env, t});
    env_batch = out_batch;
    env_free += free;
  }
  env = env.select(env.dim() - 1, 0);
  if (!env_batch)
    env = env.unsqueeze(0);
  // Order (B, S, E), with unit axes for absent modes
  const auto ps = env_free.find('s'), pe = env_free.find('e');
  if (ps == std::string::npos)
    env = env.unsqueeze(-1), env_free += 's';
  if (pe == std::string::npos)
    env = env.unsqueeze(-1), env_free += 'e';
  if (env_free.size() != 2) {
    throw utils::runtime_error(
      ctx("contract"), "Angle modes must be batched or weighted");
  }
  if (env_free[0] == 'e')
    env = env.transpose(1, 2);
  return env;
}

torch::Tensor LayoutCross::moments() const
{
  using Kind = Selector::Kind;
  auto Yw = problem_.harmonics.to(torch::kFloat64) *
            problem_.weights.to(torch::kFloat64).unsqueeze(1);
  const int64_t M = Yw.size(1);
  auto out = torch::empty({M, size_[E], size_[S]}, torch::kFloat64);
  const bool has_ga = layout_.contains(tensor::Dimension::AZIMUTHAL);
  const bool same = has_ga && layout_.core_of(tensor::Dimension::POLAR) ==
                                layout_.core_of(tensor::Dimension::AZIMUTHAL);
  for (int64_t m = 0; m < M; m++) {
    auto W = Yw.select(1, m).reshape({size_[MU], size_[GA]}).contiguous();
    std::array<Selector, 4> sel;
    if (!has_ga) {
      sel[MU] = {Kind::WEIGHT, W.select(1, 0).contiguous()};
    } else if (same) {
      sel[MU] = {Kind::WEIGHT, {}};
      sel[GA] = {Kind::WEIGHT, {}};
    } else {
      auto [u, v] = outer_factors(W, 1e-10);
      if (!u.defined()) {
        // A zero moment weight (e.g. odd harmonics) factors trivially
        if (W.abs().max().item<double>() == 0.0) {
          out[m].zero_();
          continue;
        }
        throw utils::runtime_error(ctx("moments"),
          "Splitting polar from azimuthal angle needs separable harmonics "
          "(a product quadrature)");
      }
      sel[MU] = {Kind::WEIGHT, u};
      sel[GA] = {Kind::WEIGHT, v};
    }
    out[m] = contract(sel, W)[0].t();
  }
  return out;
}

torch::Tensor LayoutCross::evaluate(const std::vector<int64_t>& directions,
  const std::vector<int64_t>& groups, const std::vector<int64_t>& space) const
{
  using Kind = Selector::Kind;
  const int64_t Ng = size_[GA];
  std::vector<int64_t> mu, ga;
  for (int64_t d : directions) {
    mu.push_back(d / Ng);
    ga.push_back(d % Ng);
  }
  std::array<Selector, 4> sel;
  sel[MU] = {Kind::BATCH, torch::tensor(mu, torch::kInt64)};
  sel[GA] = {Kind::BATCH, torch::tensor(ga, torch::kInt64)};
  sel[S] = {Kind::LIST, torch::tensor(space, torch::kInt64)};
  sel[E] = {Kind::LIST, torch::tensor(groups, torch::kInt64)};
  return contract(sel).transpose(1, 2).contiguous();
}

std::vector<int64_t> LayoutCross::ranks() const
{
  std::vector<int64_t> r {1};
  for (const auto& c : cores_)
    r.push_back(c.defined() ? c.size(2) : 0);
  return r;
}

// =================================================================
// Per-patch TTs

std::vector<torch::Tensor> LayoutCross::patch_tt(size_t p) const
{
  if (p >= problem_.patch_offsets.size())
    throw utils::runtime_error(ctx("patch_tt"), "Patch out of range");
  if (!has_tt())
    throw utils::runtime_error(ctx("patch_tt"),
      "No TT held: run() first, or set keep_tt to keep it after a solve");
  const int64_t off = problem_.patch_offsets[p];
  const int64_t end = p + 1 < problem_.patch_offsets.size()
                        ? problem_.patch_offsets[p + 1]
                        : size_[S];
  const auto& pshape = problem_.patch_shapes.at(p);
  const auto ks = layout_.core_of(tensor::Dimension::SPACE);

  // Axis labels (dimensions) of every solver core, space sliced and
  // expanded to the patch grid when the user splits it
  const bool split = !user_layout_.contains(tensor::Dimension::SPACE);
  const tensor::Dimension xyz[] = {
    tensor::Dimension::X, tensor::Dimension::Y, tensor::Dimension::Z};
  std::vector<torch::Tensor> blocks;
  std::vector<std::vector<tensor::Dimension>> labels;
  for (int64_t k = 0; k < K_; k++) {
    std::vector<int64_t> shape {cores_[k].size(0)};
    std::vector<tensor::Dimension> lab;
    for (size_t q = 0; q < core_modes_[k].size(); q++) {
      const int m = core_modes_[k][q];
      if (m == S) {
        if (split) {
          for (size_t i = 0; i < pshape.size(); i++) {
            shape.push_back(pshape[i]);
            lab.push_back(xyz[i]);
          }
        } else {
          shape.push_back(end - off);
          lab.push_back(tensor::Dimension::SPACE);
        }
      } else {
        shape.push_back(core_dims_[k][q]);
        lab.push_back(m == MU   ? tensor::Dimension::POLAR
                      : m == GA ? tensor::Dimension::AZIMUTHAL
                                : tensor::Dimension::ENERGY);
      }
    }
    shape.push_back(cores_[k].size(2));
    auto t = cores_[k];
    if (k == ks) {
      std::vector<int64_t> view {t.size(0)};
      view.insert(view.end(), core_dims_[k].begin(), core_dims_[k].end());
      view.push_back(t.size(2));
      const auto qs =
        std::find(core_modes_[k].begin(), core_modes_[k].end(), S) -
        core_modes_[k].begin();
      t = t.reshape(view).narrow(qs + 1, off, end - off);
    }
    blocks.push_back(t.reshape(shape));
    labels.push_back(std::move(lab));
  }

  // Segments: consecutive solver cores and the user cores spanning them
  std::vector<int64_t> solver_of_user;
  for (const auto& ucore : user_layout_.cores()) {
    int64_t lo = K_, hi = -1;
    for (const auto& idx : ucore) {
      const int64_t c = tensor::is_spatial(idx.dimension())
                          ? ks
                          : layout_.core_of(idx.dimension());
      lo = std::min(lo, c);
      hi = std::max(hi, c);
    }
    solver_of_user.push_back(lo);
    solver_of_user.push_back(hi);
  }
  const int64_t U = static_cast<int64_t>(user_layout_.num_cores());
  std::vector<torch::Tensor> out;
  int64_t u0 = 0;
  while (u0 < U) {
    int64_t k0 = solver_of_user[2 * u0], k1 = solver_of_user[2 * u0 + 1];
    int64_t u1 = u0;
    while (u1 + 1 < U && solver_of_user[2 * (u1 + 1)] <= k1) {
      u1++;
      k1 = std::max(k1, solver_of_user[2 * u1 + 1]);
    }
    // Contract solver cores k0..k1
    auto block = blocks[k0];
    std::vector<tensor::Dimension> lab = labels[k0];
    for (int64_t k = k0 + 1; k <= k1; k++) {
      block = torch::tensordot(block, blocks[k], {block.dim() - 1}, {0});
      lab.insert(lab.end(), labels[k].begin(), labels[k].end());
    }
    // Permute to the user order and split into the user cores
    std::vector<int64_t> perm {0}, groups;
    for (int64_t u = u0; u <= u1; u++) {
      const auto& ucore = user_layout_.core(u);
      groups.push_back(static_cast<int64_t>(ucore.size()));
      for (const auto& idx : ucore) {
        const auto pos =
          std::find(lab.begin(), lab.end(), idx.dimension()) - lab.begin();
        perm.push_back(pos + 1);
      }
    }
    perm.push_back(block.dim() - 1);
    block = block.permute(perm).contiguous();
    // TT-SVD of (r_l * n_1, n_2, ..., n_q * r_r) gives the user cores
    std::vector<int64_t> sizes;
    int64_t axis = 1;
    for (int64_t gsize : groups) {
      int64_t n = 1;
      for (int64_t i = 0; i < gsize; i++)
        n *= block.size(axis++);
      sizes.push_back(n);
    }
    const int64_t rl = block.size(0), rr = block.size(block.dim() - 1);
    auto modes = sizes;
    modes.front() *= rl;
    modes.back() *= rr;
    auto pieces = linalg::TTEngine::tt_svd(
      block.reshape(modes), 0.1 * config_.tol, max_rank());
    for (size_t i = 0; i < pieces.size(); i++) {
      auto piece = pieces[i];
      const int64_t a = i == 0 ? rl : piece.size(0);
      const int64_t b = i + 1 == pieces.size() ? rr : piece.size(2);
      out.push_back(piece.reshape({a, sizes[i], b}));
    }
    u0 = u1 + 1;
  }
  linalg::TTEngine::Tensors cores(out.begin(), out.end());
  linalg::TTEngine tt(cores);
  tt.round_(config_.tol, max_rank());
  const auto& rounded = tt.get_cores();
  return {rounded.begin(), rounded.end()};
}

} // namespace ttnte::cross
