#include "ttnte/cross/pivots.hpp"
#include <ATen/CPUGeneratorImpl.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ttnte::cross {

std::vector<int64_t> pivot_columns(const torch::Tensor& R, double eps)
{
  const int64_t k = R.size(0);
  const int64_t n = R.size(1);
  // Column-major working copy
  auto A = R.to(torch::kFloat64).t().contiguous();
  double* a = A.data_ptr<double>();
  std::vector<int64_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::vector<double> norm2(n);
  double total = 0.0;
  for (int64_t j = 0; j < n; j++) {
    double s = 0.0;
    for (int64_t i = 0; i < k; i++)
      s += a[j * k + i] * a[j * k + i];
    norm2[j] = s;
    total += s;
  }
  std::vector<int64_t> picked;
  const double stop = eps * eps * total;
  double tail = total;
  for (int64_t step = 0; step < std::min(k, n); step++) {
    if (step > 0 && tail <= stop)
      break;
    // Pivot: the column with the largest remaining norm
    int64_t best = step;
    for (int64_t j = step + 1; j < n; j++)
      if (norm2[j] > norm2[best])
        best = j;
    if (norm2[best] <= 0.0)
      break;
    if (best != step) {
      std::swap_ranges(a + step * k, a + (step + 1) * k, a + best * k);
      std::swap(norm2[step], norm2[best]);
      std::swap(perm[step], perm[best]);
    }
    picked.push_back(perm[step]);
    // Householder reflector zeroing column `step` below row `step`
    double* v = a + step * k;
    double alpha = 0.0;
    for (int64_t i = step; i < k; i++)
      alpha += v[i] * v[i];
    alpha = std::sqrt(alpha);
    if (alpha == 0.0)
      continue;
    if (v[step] > 0)
      alpha = -alpha;
    v[step] -= alpha;
    double vnorm2 = 0.0;
    for (int64_t i = step; i < k; i++)
      vnorm2 += v[i] * v[i];
    for (int64_t j = step + 1; j < n; j++) {
      double* c = a + j * k;
      double dot = 0.0;
      for (int64_t i = step; i < k; i++)
        dot += v[i] * c[i];
      const double f = 2.0 * dot / vnorm2;
      for (int64_t i = step; i < k; i++)
        c[i] -= f * v[i];
      // Remaining norm: the part below row `step`
      double s = 0.0;
      for (int64_t i = step + 1; i < k; i++)
        s += c[i] * c[i];
      norm2[j] = s;
    }
    tail = 0.0;
    for (int64_t j = step + 1; j < n; j++)
      tail += norm2[j];
  }
  if (picked.empty())
    picked.push_back(0);
  return picked;
}

torch::Tensor direction_sketch(int64_t direction, int64_t k, int64_t n)
{
  auto gen = at::detail::createCPUGenerator(
    0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(direction));
  return at::randn({k, n}, gen, torch::kFloat64);
}

namespace {

/// Scores at or below this are no improvement (squared, relative).
constexpr double NO_GAIN = 1e-28;

/// @brief Among the unpicked candidates scoring within `near` of the best,
/// the cheapest, then the best scoring.
/// @return The pick, or -1 if no unpicked candidate gains anything.
int64_t pick_near(const std::vector<double>& score,
  const std::vector<double>& cost, const std::vector<uint8_t>& picked,
  double near)
{
  const int64_t n = static_cast<int64_t>(score.size());
  double best = -1.0;
  for (int64_t u = 0; u < n; u++)
    if (!picked[u])
      best = std::max(best, score[u]);
  if (best <= NO_GAIN)
    return -1;
  int64_t pick = -1;
  for (int64_t u = 0; u < n; u++) {
    if (picked[u] || score[u] < near * best)
      continue;
    if (pick < 0 || cost[u] < cost[pick] ||
        (cost[u] == cost[pick] && score[u] > score[pick]))
      pick = u;
  }
  return pick;
}

} // namespace

void select_block_grams(const torch::Tensor& grams,
  const std::vector<int64_t>& sizes, const std::vector<double>& cost,
  std::vector<uint8_t>& picked, double cond, double near)
{
  const int64_t nk = grams.size(0);
  const int64_t r = grams.size(1);
  auto flat = grams.reshape({nk, r * r}).contiguous();
  auto trace_t = grams.diagonal(0, 1, 2).sum(1).contiguous();
  const double* trace = trace_t.data_ptr<double>();
  double total = 0.0;
  for (int64_t u = 0; u < nk; u++)
    total += trace[u];
  total = std::max(total, 1e-300);

  // With at most r candidate rows the chosen rows can never be r well-
  // conditioned ones: the loop below would add every block, one
  // eigendecomposition at a time
  int64_t all_rows = 0;
  for (int64_t u = 0; u < nk; u++)
    all_rows += sizes[u];
  if (all_rows <= r) {
    std::fill(picked.begin(), picked.end(), 1);
    return;
  }

  auto gram = torch::zeros({r, r}, torch::kFloat64);
  int64_t num_rows = 0;
  for (int64_t u = 0; u < nk; u++) {
    if (picked[u]) {
      gram += grams[u];
      num_rows += sizes[u];
    }
  }
  std::vector<double> score(nk);
  while (true) {
    if (num_rows > 0) {
      auto [ev, V] = torch::linalg_eigh(gram);
      auto s = ev.clamp_min(0.0).sqrt().flip(0);
      V = V.flip(1);
      const double smax = s[0].item<double>();
      const double smin = s[r - 1].item<double>();
      if (num_rows >= r && smin >= cond * smax)
        break;
      auto Vk = V.index_select(1, (s > cond * smax).nonzero().flatten());
      auto proj = flat.mv(Vk.mm(Vk.t()).flatten()).contiguous();
      const double* pj = proj.data_ptr<double>();
      for (int64_t u = 0; u < nk; u++)
        score[u] = std::max(trace[u] - pj[u], 0.0) / total;
    } else {
      for (int64_t u = 0; u < nk; u++)
        score[u] = trace[u] / total;
    }
    const int64_t pick = pick_near(score, cost, picked, near);
    if (pick < 0)
      break;
    picked[pick] = 1;
    num_rows += sizes[pick];
    gram += grams[pick];
  }
}

void select_single_rows(const torch::Tensor& rows,
  const std::vector<double>& cost, std::vector<uint8_t>& picked, double cond,
  double near)
{
  auto B = rows.to(torch::kFloat64).contiguous();
  const int64_t m = B.size(0), r = B.size(1);
  if (m <= r) {
    std::fill(picked.begin(), picked.end(), 1);
    return;
  }
  auto norms_t = B.square().sum(1).contiguous();
  std::vector<double> resid(
    norms_t.data_ptr<double>(), norms_t.data_ptr<double>() + m);
  double total = 0.0;
  for (double x : resid)
    total += x;
  total = std::max(total, 1e-300);
  auto Q = torch::empty({r, r}, torch::kFloat64);
  // Gram of the chosen rows, for the stopping test and, once Q is full, the
  // scores
  auto gram = torch::zeros({r, r}, torch::kFloat64);
  int64_t q = 0, num_rows = 0;

  // Extend the basis by row i (twice orthogonalized) and update residuals
  auto add = [&](int64_t i) {
    num_rows++;
    gram.addr_(B[i], B[i]);
    if (q >= r)
      return;
    auto v = B[i].clone();
    for (int pass = 0; pass < 2 && q > 0; pass++) {
      auto Qk = Q.narrow(1, 0, q);
      v -= Qk.mv(Qk.t().mv(v));
    }
    const double nv = v.norm().item<double>();
    if (nv <= 1e-12 * std::sqrt(std::max(norms_t[i].item<double>(), 1e-300)))
      return;
    v /= nv;
    Q.select(1, q).copy_(v);
    q++;
    auto c = B.mv(v).contiguous();
    const double* cp = c.data_ptr<double>();
    for (int64_t j = 0; j < m; j++)
      resid[j] = std::max(resid[j] - cp[j] * cp[j], 0.0);
  };
  for (int64_t i = 0; i < m; i++)
    if (picked[i])
      add(i);

  std::vector<double> score(m);
  while (true) {
    bool scored = false;
    if (num_rows >= r) {
      auto [ev, Z] = torch::linalg_eigh(gram); // ascending
      auto sv = ev.clamp_min(0.0).sqrt();
      const double smax = sv[r - 1].item<double>();
      if (sv[0].item<double>() >= cond * smax)
        break;
      if (q >= r) {
        // The chosen rows span everything, so every Gram-Schmidt residual is
        // zero up to round-off and cannot rank the rows. Score instead, as
        // select_block_grams() does, by each row's energy in the poorly
        // resolved directions (sigma_j <= cond sigma_max): the rows that
        // can raise sigma_min
        auto Zp = Z.index_select(1, (sv <= cond * smax).nonzero().flatten());
        auto e = B.mm(Zp).square().sum(1).contiguous();
        const double* ep = e.data_ptr<double>();
        for (int64_t i = 0; i < m; i++)
          score[i] = ep[i] / total;
        scored = true;
      }
    }
    if (!scored)
      for (int64_t i = 0; i < m; i++)
        score[i] = resid[i] / total;
    const int64_t pick = pick_near(score, cost, picked, near);
    if (pick < 0)
      break;
    picked[pick] = 1;
    add(pick);
  }
}

} // namespace ttnte::cross
