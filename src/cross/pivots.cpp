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

std::vector<int64_t> select_blocks(const torch::Tensor& B,
  const std::vector<int64_t>& keys, const std::function<double(int64_t)>& cost,
  const std::vector<int64_t>& start, double cond, double near)
{
  const int64_t m = B.size(0);
  const int64_t r = B.size(1);
  std::vector<int64_t> ukeys(keys.begin(), keys.end());
  std::sort(ukeys.begin(), ukeys.end());
  ukeys.erase(std::unique(ukeys.begin(), ukeys.end()), ukeys.end());
  const int64_t nk = static_cast<int64_t>(ukeys.size());
  std::vector<int64_t> inv(m);
  for (int64_t i = 0; i < m; i++)
    inv[i] =
      std::lower_bound(ukeys.begin(), ukeys.end(), keys[i]) - ukeys.begin();
  std::vector<double> kcost(nk);
  for (int64_t u = 0; u < nk; u++)
    kcost[u] = cost(ukeys[u]);
  std::vector<uint8_t> picked(nk, 0);
  for (int64_t key : start) {
    auto it = std::lower_bound(ukeys.begin(), ukeys.end(), key);
    if (it != ukeys.end() && *it == key)
      picked[it - ukeys.begin()] = 1;
  }
  std::vector<uint8_t> rows(m);
  for (int64_t i = 0; i < m; i++)
    rows[i] = picked[inv[i]];

  auto Bd = B.to(torch::kFloat64).contiguous();
  auto row_list = [&](const std::function<bool(int64_t)>& take) {
    std::vector<int64_t> idx;
    for (int64_t i = 0; i < m; i++)
      if (take(i))
        idx.push_back(i);
    return idx;
  };
  auto as_tensor = [](const std::vector<int64_t>& v) {
    return torch::tensor(v, torch::kInt64);
  };
  auto chosen = row_list([&](int64_t i) { return rows[i] != 0; });
  auto gram = torch::zeros({r, r}, torch::kFloat64);
  if (!chosen.empty()) {
    auto Bs = Bd.index_select(0, as_tensor(chosen));
    gram = Bs.t().mm(Bs);
  }
  auto norms_t = Bd.square().sum(1);
  const double* norms = norms_t.data_ptr<double>();
  const double total = std::max(norms_t.sum().item<double>(), 1e-300);
  int64_t num_rows =
    std::count_if(rows.begin(), rows.end(), [](uint8_t x) { return x; });

  while (true) {
    torch::Tensor resid;
    if (num_rows > 0) {
      auto [ev, V] = torch::linalg_eigh(gram);
      auto s = ev.clamp_min(0.0).sqrt().flip(0);
      V = V.flip(1);
      const double smax = s[0].item<double>();
      const double smin = s[r - 1].item<double>();
      if (num_rows >= r && smin >= cond * smax)
        break;
      auto keep = (s > cond * smax).nonzero().flatten();
      auto P = Bd.mm(V.index_select(1, keep));
      resid = norms_t - P.square().sum(1);
    } else {
      resid = norms_t.clone();
    }
    resid = resid.clamp_min(0.0).contiguous();
    const double* rs = resid.data_ptr<double>();
    std::vector<double> score(nk, 0.0);
    for (int64_t i = 0; i < m; i++)
      score[inv[i]] += rs[i] / total;
    double best = -1.0;
    for (int64_t u = 0; u < nk; u++) {
      if (picked[u])
        score[u] = -1.0;
      best = std::max(best, score[u]);
    }
    if (best <= 1e-28)
      break;
    // Among blocks within `near` of the best, the cheapest, then the best
    int64_t pick = -1;
    for (int64_t u = 0; u < nk; u++) {
      if (score[u] < near * best)
        continue;
      if (pick < 0 || kcost[u] < kcost[pick] ||
          (kcost[u] == kcost[pick] && score[u] > score[pick]))
        pick = u;
    }
    picked[pick] = 1;
    auto fresh = row_list([&](int64_t i) { return inv[i] == pick; });
    for (int64_t i : fresh)
      rows[i] = 1;
    num_rows += static_cast<int64_t>(fresh.size());
    auto Bn = Bd.index_select(0, as_tensor(fresh));
    gram += Bn.t().mm(Bn);
  }
  std::vector<int64_t> out;
  for (int64_t i = 0; i < m; i++)
    if (rows[i])
      out.push_back(i);
  return out;
}

} // namespace ttnte::cross
