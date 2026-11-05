// LayoutCross: Index sets and block selection.
#include "cross/layout_cross_detail.hpp"
#include "ttnte/cross/layout_cross.hpp"

namespace ttnte::cross {

using namespace detail;

// =================================================================
// Index sets

LayoutCross::IndexSet LayoutCross::candidates(
  size_t k, bool left, const std::vector<int64_t>& rows) const
{
  // Rows are (i, j) with i in left_[k], or (j, l) with l in right_[k + 1]
  const auto& side = left ? left_[k] : right_[k + 1];
  const int64_t inner = left ? n_[k] : side.size();
  IndexSet out;
  out.comp.reserve(rows.size() * 4);
  for (int64_t c : rows) {
    const int64_t j = left ? c % inner : c / inner;
    const int64_t i = left ? c / inner : c % inner;
    int64_t comp[4];
    std::copy(&side.comp[i * 4], &side.comp[i * 4] + 4, comp);
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
      const double* const* row = &slot_[(m * Ng + a) * G];
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

LayoutCross::CandidateBlocks LayoutCross::candidate_blocks(
  size_t k, bool left, const torch::Tensor& core) const
{
  using Group = CandidateBlocks::Group;
  CandidateBlocks cb;
  const IndexSet& side = left ? left_[k] : right_[k + 1];
  cb.ms = side.size();
  cb.nk = n_[k];
  const int64_t ms = cb.ms, nk = cb.nk;
  if (left) {
    cb.Sv = B_[k].defined() ? B_[k] : torch::eye(ms, torch::kFloat64);
    cb.K = core.permute({1, 0, 2});
  } else {
    cb.Sv = C_[k + 1].defined() ? C_[k + 1] : torch::eye(ms, torch::kFloat64);
    cb.K = core.permute({1, 2, 0});
  }
  // K stays a view of the core: only the multi-row path gathers from it
  cb.Sv = cb.Sv.contiguous();
  cb.q = cb.Sv.size(1);
  cb.r = cb.K.size(2);
  const int64_t q = cb.q, r = cb.r;
  if (cb.K.size(0) != nk || cb.K.size(1) != q)
    throw utils::runtime_error(
      ctx("candidate_blocks"), "Core does not match the side basis");

  // Side rows grouped by their (non-space) key parts
  {
    std::unordered_map<int64_t, size_t> where;
    for (int64_t a = 0; a < ms; a++) {
      std::array<int64_t, 4> c;
      std::copy(&side.comp[a * 4], &side.comp[a * 4] + 4, c.begin());
      c[S] = -1;
      auto [it, fresh] = where.emplace(row_key(c.data()), cb.side.size());
      if (fresh)
        cb.side.push_back({c, {}});
      cb.side[it->second].members.push_back(a);
    }
  }
  // Core values grouped arithmetically rather than through a hash over all
  // nk values: with space in the core, value j = (o Ns + s) inner + ci and
  // group (o, ci) holds its Ns values; without, each value is its own group
  // (same groups, same order as grouping by key)
  {
    const auto [outer, inner] = space_split(k);
    if (outer == 0) {
      cb.core.resize(nk);
      for (int64_t j = 0; j < nk; j++) {
        std::array<int64_t, 4> c {-1, -1, -1, -1};
        decode(k, j, c.data());
        cb.core[j] = {c, {j}};
      }
    } else {
      const int64_t Ns = size_[S];
      cb.core.reserve(outer * inner);
      for (int64_t o = 0; o < outer; o++)
        for (int64_t ci = 0; ci < inner; ci++) {
          Group g;
          std::fill(g.comp.begin(), g.comp.end(), -1);
          decode(k, o * Ns * inner + ci, g.comp.data());
          g.comp[S] = -1;
          g.members.resize(Ns);
          for (int64_t s = 0; s < Ns; s++)
            g.members[s] = (o * Ns + s) * inner + ci;
          cb.core.push_back(std::move(g));
        }
    }
  }

  // One block per (side group, core group)
  const int64_t nA = static_cast<int64_t>(cb.side.size());
  const int64_t nJ = static_cast<int64_t>(cb.core.size());
  for (const auto& g : cb.side)
    cb.single = cb.single && g.members.size() == 1;
  cb.single_rows = cb.single;
  for (const auto& g : cb.core)
    cb.single_rows = cb.single_rows && g.members.size() == 1;
  cb.J_index.resize(nJ);
  for (int64_t J = 0; J < nJ; J++)
    cb.J_index[J] = torch::tensor(cb.core[J].members, torch::kInt64);
  cb.keys.resize(nA * nJ);
  for (int64_t bk = 0; bk < nA * nJ; bk++) {
    int64_t c[4];
    std::copy(cb.side[bk / nJ].comp.begin(), cb.side[bk / nJ].comp.end(), c);
    merge(c, cb.core[bk % nJ].comp.data());
    cb.keys[bk] = row_key(c);
  }
  // Single-row side groups (the common case): the candidate rows
  // T[a, j] = S_a K_j for a in group order, by one GEMM on the core's own
  // memory, no copy of the core
  if (cb.single) {
    auto A_order = torch::empty({ms}, torch::kInt64);
    auto* o = A_order.data_ptr<int64_t>();
    for (int64_t A = 0; A < nA; A++)
      o[A] = cb.side[A].members[0];
    auto SA = cb.Sv.index_select(0, A_order);
    const auto c = core.contiguous();
    if (left)
      cb.T = SA.mm(c.view({q, nk * r})).view({ms, nk, r});
    else
      cb.T = c.view({r * nk, q})
               .mm(SA.t())
               .view({r, nk, ms})
               .permute({2, 1, 0})
               .contiguous();
  }
  return cb;
}

torch::Tensor LayoutCross::block_grams(
  const CandidateBlocks& cb, bool factored) const
{
  const int64_t nA = static_cast<int64_t>(cb.side.size());
  const int64_t nJ = static_cast<int64_t>(cb.core.size());
  const int64_t q = cb.q, r = cb.r;
  auto grams = torch::empty({nA, nJ, r, r}, torch::kFloat64);
  if (cb.single && !factored) {
    // Equal, contiguous core groups (space last in the core): every block
    // Gram from one batched product instead of one per block
    const int64_t gsize = static_cast<int64_t>(cb.core[0].members.size());
    bool blocked = nJ * gsize == cb.nk;
    for (int64_t J = 0; blocked && J < nJ; J++) {
      const auto& mem = cb.core[J].members;
      blocked = static_cast<int64_t>(mem.size()) == gsize;
      for (int64_t i = 0; blocked && i < gsize; i++)
        blocked = mem[i] == J * gsize + i;
    }
    if (blocked) {
      auto T3 = cb.T.view({nA * nJ, gsize, r});
      grams.copy_(torch::bmm(T3.transpose(1, 2), T3).view({nA, nJ, r, r}));
    } else {
      for (int64_t J = 0; J < nJ; J++) {
        auto TJ = cb.T.index_select(1, cb.J_index[J]); // (nA, |J|, r)
        grams.select(1, J).copy_(torch::bmm(TJ.transpose(1, 2), TJ));
      }
    }
  } else {
    // Multi-row side groups: H_A = S_A^T S_A once per group, so the cost
    // does not grow with the group's rows
    auto H = torch::empty({nA, q, q}, torch::kFloat64);
    for (int64_t A = 0; A < nA; A++) {
      auto SA =
        cb.Sv.index_select(0, torch::tensor(cb.side[A].members, torch::kInt64));
      H[A].copy_(SA.t().mm(SA));
    }
    for (int64_t J = 0; J < nJ; J++) {
      auto KJ = cb.K.index_select(0, cb.J_index[J]); // (|J|, q, r)
      grams.select(1, J).copy_(torch::einsum("jqr,aqp,jps->ars", {KJ, H, KJ}));
    }
  }
  return grams.reshape({nA * nJ, r, r});
}

std::tuple<std::vector<int64_t>, torch::Tensor, torch::Tensor>
LayoutCross::select_candidates(size_t k, bool left, const torch::Tensor& core,
  const std::vector<int64_t>& keep)
{
  const auto cb = candidate_blocks(k, left, core);
  const auto& side_groups = cb.side;
  const auto& core_groups = cb.core;
  const auto& keys = cb.keys;
  const int64_t ms = cb.ms, nk = cb.nk, r = cb.r;
  const int64_t nA = static_cast<int64_t>(side_groups.size());
  const int64_t nJ = static_cast<int64_t>(core_groups.size());
  const int64_t nb = nA * nJ;
  // Block Grams, unless every block is one row (selected on the rows)
  torch::Tensor grams;
  if (!cb.single_rows)
    grams = block_grams(cb, factored_grams_);
  std::vector<int64_t> sizes(nb);
  std::vector<double> cost(nb);
  for (int64_t bk = 0; bk < nb; bk++) {
    sizes[bk] = static_cast<int64_t>(side_groups[bk / nJ].members.size() *
                                     core_groups[bk % nJ].members.size());
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
  if (cb.single_rows) {
    // One row per block: greedy Gram-Schmidt on the rows themselves
    std::vector<int64_t> core_index(nJ);
    for (int64_t J = 0; J < nJ; J++)
      core_index[J] = core_groups[J].members[0];
    auto rows_t = cb.T.index_select(1, torch::tensor(core_index, torch::kInt64))
                    .reshape({-1, r});
    select_single_rows(
      rows_t, cost, picked, config_.select_cond, config_.select_near);
  } else {
    select_block_grams(
      grams, sizes, cost, picked, config_.select_cond, config_.select_near);
  }

  // The chosen rows, ascending, and the new basis there
  std::vector<int64_t> rows;
  std::vector<torch::Tensor> parts;
  for (int64_t bk = 0; bk < nb; bk++) {
    if (!picked[bk])
      continue;
    const int64_t A = bk / nJ, J = bk % nJ;
    for (int64_t a : side_groups[A].members)
      for (int64_t j : core_groups[J].members)
        rows.push_back(left ? a * nk + j : j * ms + a);
    torch::Tensor part;
    if (cb.single) {
      part = cb.T.narrow(0, A, 1).index_select(1, cb.J_index[J]);
    } else {
      auto SA = cb.Sv.index_select(
        0, torch::tensor(side_groups[A].members, torch::kInt64));
      part =
        torch::einsum("aq,jqr->ajr", {SA, cb.K.index_select(0, cb.J_index[J])});
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
  // B^T B of the chosen rows is the sum of the chosen blocks' Grams
  torch::Tensor gram;
  if (!cb.single_rows) {
    std::vector<int64_t> chosen;
    for (int64_t bk = 0; bk < nb; bk++)
      if (picked[bk])
        chosen.push_back(bk);
    gram = grams.index_select(0, torch::tensor(chosen, torch::kInt64)).sum(0);
  }
  return {sorted, basis, gram};
}

void LayoutCross::grow_from_residual(const torch::Tensor& col, double total,
  size_t k, uint32_t mask, const IndexSet& current)
{
  // col[j]: the residual energy of the last core's fit at value j of core
  // k's mode, summed over the rows of the left set
  auto c = col.contiguous();
  const double* p = c.data_ptr<double>();
  const int64_t n = c.numel();
  double sum = 0.0;
  for (int64_t v = 0; v < n; v++)
    sum += p[v];
  // The fit is already within tolerance: nothing for the bond to gain
  if (std::sqrt(sum) <= config_.tol * total)
    return;

  // Residual energy per block: values that differ only in space (space in
  // core k) share one key and are added together
  auto current_keys = set_keys(current);
  std::sort(current_keys.begin(), current_keys.end());
  current_keys.erase(
    std::unique(current_keys.begin(), current_keys.end()), current_keys.end());
  auto& grown = grow_[mask];
  std::unordered_map<int64_t, double> energy;
  for (int64_t v = 0; v < n; v++) {
    if (p[v] <= 0.0)
      continue;
    int64_t comp[4] = {-1, -1, -1, -1};
    decode(k, v, comp);
    energy[row_key(comp)] += p[v];
  }
  // Only blocks not yet in the set, nor grown in an earlier pass, are
  // candidates: re-adding one would spend a slot without changing the set
  std::vector<std::pair<double, int64_t>> fresh;
  for (const auto& [key, e] : energy)
    if (!std::binary_search(current_keys.begin(), current_keys.end(), key) &&
        std::find(grown.begin(), grown.end(), key) == grown.end())
      fresh.push_back({e, key});
  // Largest residual first (ties by key, so the order is deterministic)
  std::sort(fresh.begin(), fresh.end(), [](const auto& a, const auto& b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  });

  // The fewest new blocks holding growth_share of the residual energy...
  int64_t need = 0;
  for (double acc = 0.0; need < static_cast<int64_t>(fresh.size());) {
    acc += fresh[need++].first;
    if (acc >= config_.growth_share * sum)
      break;
  }
  // ... but at most growth_max_blocks or half the set's blocks, whichever
  // is larger, so one pass cannot much more than double the set
  const int64_t cap = std::max<int64_t>(
    config_.growth_max_blocks, static_cast<int64_t>(current_keys.size()) / 2);
  const int64_t take = std::min(need, cap);
  // The next selection of these cores starts from them (select_candidates)
  for (int64_t i = 0; i < take; i++)
    grown.push_back(fresh[i].second);
}

} // namespace ttnte::cross
