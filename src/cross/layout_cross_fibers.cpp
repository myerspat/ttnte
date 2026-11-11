// LayoutCross: Requesting fibers from the store, superblocks, unweighting.
#include "cross/layout_cross_detail.hpp"
#include "ttnte/cross/layout_cross.hpp"

namespace ttnte::cross {

using namespace detail;

void LayoutCross::ensure(std::vector<int64_t> pairs, const char* phase)
{
  if (std::strcmp(phase, "check") != 0)
    for (int64_t p : pairs)
      fitted_[p] = 1;
  store_->ensure(std::move(pairs), phase);
}

// =================================================================
// Superblocks

torch::Tensor LayoutCross::superblock(size_t k, bool raw)
{
  auto t0 = Clock::now();
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
  if (holder == 1)
    std::tie(outer, inner) = space_split(k);

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
      for (int64_t t = a0; t < a1; t++) {
        const int64_t i = t / nk, j = t % nk;
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
    }
  };
  const int64_t range = holder == 0   ? static_cast<int64_t>(groups.size())
                        : holder == 1 ? ml
                                      : ml * nk;

  // Pass 1: the pairs needed, swept (timed as sweep and weight)
  std::vector<int64_t> pairs;
  visit(
    [&](int64_t pair, auto&&) {
      if (!swept(pair))
        pairs.push_back(pair);
    },
    0, range);
  timings_["gather"] += elapsed(t0);
  ensure(std::move(pairs), "superblock");

  // Pass 2: copy the samples
  t0 = Clock::now();
  auto out_t = torch::empty({ml, nk, mr}, torch::kFloat64);
  double* out = out_t.data_ptr<double>();
  if (holder == 1) {
    // Space in the core: for each (left row, outer index, inner index) the
    // mr fibers fill an (Ns, mr) block with stride inner * mr between its
    // rows; write it row by row (contiguous in l), reading the fibers side
    // by side, in parallel over (left row, outer index)
    at::parallel_for(0, ml * outer, 1, [&](int64_t a0, int64_t a1) {
      std::vector<const double*> f(mr);
      int64_t comp[4];
      for (int64_t t = a0; t < a1; t++) {
        const int64_t i = t / outer, o = t % outer;
        for (int64_t ci = 0; ci < inner; ci++) {
          for (int64_t l = 0; l < mr; l++) {
            std::copy(&L.comp[i * 4], &L.comp[i * 4] + 4, comp);
            decode(k, o * Ns * inner + ci, comp);
            merge(comp, &R.comp[l * 4]);
            f[l] = fiber(pair_of(comp));
          }
          double* dst = out + (i * nk + o * Ns * inner + ci) * mr;
          for (int64_t s = 0; s < Ns; s++) {
            double* row = dst + s * inner * mr;
            for (int64_t l = 0; l < mr; l++)
              row[l] = f[l][s];
          }
        }
      }
    });
  } else {
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
  }
  timings_["gather"] += elapsed(t0);
  if (raw)
    return out_t;
  t0 = Clock::now();
  const bool left = PL_[k].defined(), right = PR_[k + 1].defined();
  // With both projections, the first one shrinks the superblock: start with
  // the one that leaves the fewer multiplies for the second
  // (ml n mr rl + rl n mr rr against ml n mr rr + ml n rr rl)
  bool left_first = true;
  if (left && right) {
    const double ml = out_t.size(0), mr = out_t.size(2);
    const double rl = PL_[k].size(0), rr = PR_[k + 1].size(1);
    left_first = ml * mr * rl + rl * mr * rr <= ml * mr * rr + ml * rr * rl;
  }
  if (left && left_first)
    out_t = torch::tensordot(PL_[k], out_t, {1}, {0});
  if (right)
    out_t = torch::tensordot(out_t, PR_[k + 1], {2}, {0});
  if (left && !left_first)
    out_t = torch::tensordot(PL_[k], out_t, {1}, {0});
  timings_["project"] += elapsed(t0);
  return out_t;
}

void LayoutCross::unweight()
{
  const auto [angle, energy, space] = store_->weighting();
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
      auto ew = torch::tensor(store_->energy_weights(), torch::kFloat64);
      scale_axis(ae, ew.reciprocal());
    }
    const int64_t as = axis_of(S);
    if (space && as >= 0) {
      auto moved = t.movedim(as, -1).contiguous();
      auto flat = moved.reshape({-1, size_[S]});
      double* data = flat.data_ptr<double>();
      const int64_t rows = flat.size(0);
      // As in ensure(): parallel over chunks of rows, every patch's solve
      // serial inside, rather than one threaded small solve per patch
      at::parallel_for(0, rows, row_grain(rows), [&](int64_t r0, int64_t r1) {
        apply_mass(data + r0 * size_[S], r1 - r0, true);
      });
      t = moved.movedim(-1, as);
    }
    cores_[k] =
      t.reshape({cores_[k].size(0), n_[k], cores_[k].size(2)}).contiguous();
  }
}

} // namespace ttnte::cross
