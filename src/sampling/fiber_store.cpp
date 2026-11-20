// FiberStore: sweeping, weighting and storing the fibers of one sampled
// transport solve.
#include "ttnte/sampling/fiber_store.hpp"
#include "sampling/sampling_detail.hpp"
#include "ttnte/linalg/banded_cholesky.hpp"
#include "ttnte/linalg/lapack.hpp"
#include "ttnte/linalg/matrix_ops.hpp"
#include "ttnte/utils/exception.hpp"
#include <ATen/Parallel.h>
#include <algorithm>
#include <c10/core/CPUAllocator.h>
#include <c10/util/ParallelGuard.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>

namespace ttnte::sampling {

using namespace detail;

FiberStore::FiberStore(SamplingProblem problem, Weighting weighting)
  : problem_(std::move(problem))
{
  const auto& P = problem_;
  num_directions_ = P.num_polar * P.num_azimuthal;
  num_groups_ = P.num_groups;
  num_space_ = P.num_space;
  // The fiber layout: weighting writes through raw pointers at these
  // offsets, so they must tile the fiber exactly
  if (!P.patch_offsets.empty()) {
    if (P.patch_offsets.size() != P.patch_shapes.size() ||
        P.patch_offsets.front() != 0) {
      throw utils::runtime_error(ctx("FiberStore"),
        "patch_offsets must start at 0 with one shape per patch");
    }
    if (!P.mass_bands.empty() && P.mass_bands.size() != P.patch_offsets.size())
      throw utils::runtime_error(
        ctx("FiberStore"), "mass_bands must have one band per patch");
    for (size_t q = 0; q < P.patch_offsets.size(); q++) {
      const int64_t end =
        q + 1 < P.patch_offsets.size() ? P.patch_offsets[q + 1] : P.num_space;
      int64_t n = 1;
      for (int64_t d : P.patch_shapes[q])
        n *= d;
      if (end - P.patch_offsets[q] != n) {
        throw utils::runtime_error(
          ctx("FiberStore"), "Patch " + std::to_string(q) +
                               ": offsets and control-point shape disagree");
      }
      if (!P.mass_bands.empty() &&
          (P.mass_bands[q].dim() != 2 || P.mass_bands[q].size(0) != n)) {
        throw utils::runtime_error(
          ctx("FiberStore"), "Patch " + std::to_string(q) +
                               ": its mass band needs one row per dof");
      }
    }
  }

  // Weights of the samples
  angle_weight_.assign(num_directions_, 1.0);
  energy_weight_.assign(num_groups_, 1.0);
  weighting_.angle = weighting.angle;
  if (weighting.angle) {
    if (P.weights.dim() != 1 || P.weights.numel() != num_directions_)
      throw utils::runtime_error(
        ctx("FiberStore"), "weights must have one entry per direction");
    auto w = P.weights.to(torch::kFloat64).contiguous();
    const double* wp = w.data_ptr<double>();
    for (int64_t d = 0; d < num_directions_; d++)
      angle_weight_[d] = std::sqrt(wp[d]);
  }
  if (weighting.energy && !P.lethargy_widths.empty()) {
    if (static_cast<int64_t>(P.lethargy_widths.size()) != num_groups_) {
      throw utils::runtime_error(
        ctx("FiberStore"), "lethargy_widths must have one entry per group");
    }
    for (int64_t g = 0; g < num_groups_; g++) {
      if (!(P.lethargy_widths[g] > 0.0))
        throw utils::runtime_error(
          ctx("FiberStore"), "lethargy_widths must be positive");
      energy_weight_[g] = 1.0 / std::sqrt(P.lethargy_widths[g]);
    }
    weighting_.energy = true;
  }
  if (weighting.space) {
    for (const auto& band : P.mass_bands) {
      auto factor = band.to(torch::kFloat64).clone().contiguous();
      const int64_t kd = factor.size(1) - 1;
      if (linalg::banded::cholesky_factor(
            factor.data_ptr<double>(), factor.size(0), kd) != 0) {
        throw utils::runtime_error(
          ctx("FiberStore"), "A patch mass matrix is not positive definite");
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
        ctx("FiberStore"), "One mass band per patch is required");
    }
    weighting_.space = !mass_factors_.empty();
    // Group the dense factors by size
    std::map<int64_t, std::vector<size_t>> by_size;
    for (size_t q = 0; q < mass_dense_.size(); q++) {
      if (mass_dense_[q].defined())
        by_size[mass_dense_[q].size(0)].push_back(q);
      else
        banded_patches_.push_back(q);
    }
    for (const auto& [n, patches] : by_size) {
      std::vector<torch::Tensor> factors, columns;
      for (size_t q : patches) {
        factors.push_back(mass_dense_[q]);
        columns.push_back(torch::arange(n, torch::kInt64) + P.patch_offsets[q]);
      }
      dense_groups_.push_back(
        {torch::stack(factors).contiguous(), torch::cat(columns)});
    }
  }
  slot_.assign(num_directions_ * num_groups_, nullptr);
}

void FiberStore::clear()
{
  blocks_.clear();
  slot_.assign(num_directions_ * num_groups_, nullptr);
  num_swept_ = 0;
  swept_sq_ = 0.0;
  sweep_seconds_ = 0.0;
}

void FiberStore::release()
{
  blocks_.clear();
  std::fill(slot_.begin(), slot_.end(), nullptr);
}

void FiberStore::ensure(std::vector<int64_t> pairs, const char* phase)
{
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                [&](int64_t p) { return swept(p); }),
    pairs.end());
  if (pairs.empty())
    return;
  const int64_t G = num_groups_;
  const int64_t Ns = num_space_;
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

  // A new block per request: the earlier fibers are never moved
  const int64_t count = static_cast<int64_t>(pairs.size());
  blocks_.push_back(c10::GetCPUAllocator()->allocate(
    static_cast<size_t>(count * Ns) * sizeof(double)));
  double* block = static_cast<double*>(blocks_.back().get());
  // The pairs in the order of the output rows (block by block,
  // direction-major, then group)
  std::vector<int64_t> new_pairs;
  new_pairs.reserve(count);
  for (const auto& blk : blocks)
    for (int64_t d : blk.directions)
      for (int64_t g : blk.groups)
        new_pairs.push_back(d * G + g);

  // Weight finished rows: R_P per patch (BLAS-3 where the dense factor is
  // held), then sqrt(w_d) / sqrt(du_g). The oracle calls this as fibers
  // finish, inside its tasks (serial there), so it overlaps the rest of the
  // sweep; called on many rows at once it runs in parallel over chunks
  std::atomic<int64_t> weight_ns {0};
  auto weigh = [&](int64_t r0, int64_t n) {
    const auto w0 = Clock::now();
    // Small calls come from the sweep's worker threads (one task family's
    // rows): serial there, rather than an OpenMP team per worker
    const int64_t grain = loop_grain(n);
    at::parallel_for(0, n, grain, [&](int64_t b0, int64_t b1) {
      apply_mass(block + (r0 + b0) * Ns, b1 - b0, false);
      for (int64_t i = b0; i < b1; i++) {
        const int64_t p = new_pairs[r0 + i];
        const double scale = angle_weight_[p / G] * energy_weight_[p % G];
        double* f = block + (r0 + i) * Ns;
        for (int64_t s = 0; s < Ns; s++)
          f[s] *= scale;
      }
    });
    weight_ns +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - w0)
        .count();
  };

  auto t0 = Clock::now();
  problem_.sweep(blocks, block, weigh);
  const double swept = elapsed(t0);
  sweep_seconds_ += swept;
  timings_["sweep"] += swept;
  timings_["weight_cpu"] += 1e-9 * static_cast<double>(weight_ns.load());
  sweep_calls_.push_back({phase, static_cast<int64_t>(pairs.size()),
    static_cast<int64_t>(blocks.size()),
    static_cast<int64_t>(by_direction.size()), swept});

  for (int64_t i = 0; i < count; i++) {
    slot_[new_pairs[i]] = block + i * Ns;
  }
  num_swept_ += count;
  double sq = 0.0;
  for (int64_t i = 0; i < count * Ns; i++)
    sq += block[i] * block[i];
  swept_sq_ += sq;
}

void FiberStore::apply_mass(double* rows, int64_t count, bool inverse) const
{
  const int64_t Ns = num_space_;
  auto t = torch::from_blob(rows, {count, Ns}, torch::kFloat64);
  // Dense factors, a group of equal-size patches at once: rows are fibers,
  // so f R^T applies R, and solving R^T on the right (R from the left on
  // the transposed rows) applies R^{-1}
  for (const auto& grp : dense_groups_) {
    const int64_t B = grp.factors.size(0), n = grp.factors.size(1);
    const auto part = t.index_select(1, grp.columns)
                        .view({count, B, n})
                        .transpose(0, 1); // (B, count, n)
    const auto out = inverse ? torch::linalg_solve_triangular(grp.factors,
                                 part.transpose(1, 2), /*upper=*/true)
                                 .transpose(1, 2)
                             : torch::matmul(part, grp.factors.transpose(1, 2));
    t.index_copy_(1, grp.columns, out.transpose(0, 1).reshape({count, B * n}));
  }
  for (size_t q : banded_patches_) {
    const int64_t off = problem_.patch_offsets[q];
    const int64_t n = mass_factors_[q].size(0);
    const double* factor = mass_factors_[q].data_ptr<double>();
    for (int64_t i = 0; i < count; i++) {
      double* f = rows + i * Ns + off;
      if (inverse)
        linalg::banded::cholesky_solve_upper(factor, n, mass_kd_[q], f);
      else
        linalg::banded::cholesky_apply_upper(factor, n, mass_kd_[q], f);
    }
  }
}

void FiberStore::apply_dual_mass(double* rows, int64_t count) const
{
  const int64_t Ns = num_space_;
  auto t = torch::from_blob(rows, {count, Ns}, torch::kFloat64);
  // Serial, as documented, also when the caller's loop runs as one chunk on
  // its own thread (the greedy's few candidates per iteration): otherwise
  // MKL starts a thread team for each small triangular solve (Galerkin on 8
  // threads: residual norms 5x slower than on 2)
  linalg::lapack::SingleThreaded single;
  c10::ParallelGuard serial(true);
  // Dense factors by group: rows are functionals f, R^{-T} f^T = (f
  // R^{-1})^T, solving R on the right
  for (const auto& grp : dense_groups_) {
    const int64_t B = grp.factors.size(0), n = grp.factors.size(1);
    const auto part = t.index_select(1, grp.columns)
                        .view({count, B, n})
                        .transpose(0, 1); // (B, count, n)
    const auto out = torch::linalg_solve_triangular(
      grp.factors, part, /*upper=*/true, /*left=*/false);
    t.index_copy_(1, grp.columns, out.transpose(0, 1).reshape({count, B * n}));
  }
  for (size_t q : banded_patches_) {
    const int64_t off = problem_.patch_offsets[q];
    const int64_t n = mass_factors_[q].size(0);
    // R^{-T} = L^{-1} with the band factor M = L L^T
    const double* factor = mass_factors_[q].data_ptr<double>();
    for (int64_t i = 0; i < count; i++)
      linalg::banded::cholesky_solve_lower(
        factor, n, mass_kd_[q], rows + i * Ns + off);
  }
}

torch::Tensor FiberStore::residual_norms(
  const std::vector<int64_t>& directions, const torch::Tensor& residuals) const
{
  const int64_t D = static_cast<int64_t>(directions.size());
  const int64_t G = num_groups_, Ns = num_space_;
  if (residuals.dim() != 3 || residuals.size(0) != D ||
      residuals.size(1) != G || residuals.size(2) != Ns)
    throw utils::runtime_error(ctx("residual_norms"),
      "residuals must be (len(directions), num_groups, num_space)");
  for (int64_t d : directions)
    if (d < 0 || d >= num_directions_)
      throw utils::runtime_error(
        ctx("residual_norms"), "Direction out of range");
  // Weighted in a copy, row (d, g) by row, in parallel over rows
  auto work = residuals.to(torch::kFloat64).contiguous().clone();
  double* data = work.data_ptr<double>();
  const int64_t rows = D * G;
  std::vector<double> sq(static_cast<size_t>(rows), 0.0);
  const int64_t grain = loop_grain(rows);
  at::parallel_for(0, rows, grain, [&](int64_t r0, int64_t r1) {
    apply_dual_mass(data + r0 * Ns, r1 - r0);
    for (int64_t i = r0; i < r1; i++) {
      const double scale =
        angle_weight_[directions[i / G]] * energy_weight_[i % G];
      const double* f = data + i * Ns;
      double s = 0.0;
      for (int64_t k = 0; k < Ns; k++)
        s += f[k] * f[k];
      sq[i] = scale * scale * s;
    }
  });
  auto norms = torch::empty({D}, torch::kFloat64);
  double* out = norms.data_ptr<double>();
  for (int64_t d = 0; d < D; d++) {
    double s = 0.0;
    for (int64_t g = 0; g < G; g++)
      s += sq[d * G + g];
    out[d] = std::sqrt(s);
  }
  return norms;
}

torch::Tensor FiberStore::pair_residual_norms(
  const std::vector<int64_t>& pairs, const torch::Tensor& residuals) const
{
  const int64_t U = static_cast<int64_t>(pairs.size());
  const int64_t G = num_groups_, Ns = num_space_;
  if (!((residuals.dim() == 2 && residuals.size(0) == U &&
          residuals.size(1) == Ns) ||
        (residuals.dim() == 3 && residuals.size(0) == U &&
          residuals.size(1) == 1 && residuals.size(2) == Ns)))
    throw utils::runtime_error(
      ctx("pair_residual_norms"), "residuals must be (len(pairs), num_space)");
  for (int64_t u : pairs)
    if (u < 0 || u >= num_directions_ * G)
      throw utils::runtime_error(
        ctx("pair_residual_norms"), "Pair out of range");
  // Weighted in a copy, row by row, in parallel over rows
  auto work =
    residuals.to(torch::kFloat64).reshape({U, Ns}).contiguous().clone();
  double* data = work.data_ptr<double>();
  auto norms = torch::empty({U}, torch::kFloat64);
  double* out = norms.data_ptr<double>();
  const int64_t grain = loop_grain(U);
  at::parallel_for(0, U, grain, [&](int64_t r0, int64_t r1) {
    apply_dual_mass(data + r0 * Ns, r1 - r0);
    for (int64_t i = r0; i < r1; i++) {
      const int64_t u = pairs[i];
      const double scale = angle_weight_[u / G] * energy_weight_[u % G];
      const double* f = data + i * Ns;
      double sq = 0.0;
      for (int64_t k = 0; k < Ns; k++)
        sq += f[k] * f[k];
      out[i] = scale * std::sqrt(sq);
    }
  });
  return norms;
}

} // namespace ttnte::sampling
