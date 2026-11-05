// LayoutCross: Contractions, moments, evaluation and per-patch TTs.
#include "cross/layout_cross_detail.hpp"
#include "ttnte/cross/layout_cross.hpp"

namespace ttnte::cross {

using namespace detail;

// =================================================================
// Contractions

torch::Tensor LayoutCross::contract(const std::array<Selector, 4>& select,
  const torch::Tensor& angle_matrix) const
{
  using Kind = Selector::Kind;
  if (!has_tt())
    throw utils::runtime_error(ctx("contract"),
      "No TT held: run() first, or set keep_angular_flux to "
      "keep it after a solve");
  const char letter[4] = {'m', 'g', 's', 'e'};
  int64_t core_of_mode[4] = {-1, -1, -1, -1};
  for (int64_t k = 0; k < K_; k++)
    for (int m : core_modes_[k])
      core_of_mode[m] = k;
  const bool joint = select[MU].kind == Kind::WEIGHT &&
                     select[GA].kind == Kind::WEIGHT && core_of_mode[GA] >= 0 &&
                     core_of_mode[MU] == core_of_mode[GA];
  for (int m = 0; m < 4; m++) {
    const auto& sel = select[m];
    const bool angle = m == MU || m == GA;
    if ((angle && sel.kind != Kind::BATCH && sel.kind != Kind::WEIGHT) ||
        (m == S && sel.kind != Kind::ALL && sel.kind != Kind::LIST)) {
      throw utils::runtime_error(ctx("contract"),
        "Angle modes take BATCH or WEIGHT selectors, space ALL or LIST");
    }
    if (sel.kind == Kind::ALL || (sel.kind == Kind::WEIGHT && joint))
      continue;
    const auto& d = sel.data;
    const bool indices = sel.kind != Kind::WEIGHT;
    if (!d.defined() || d.dim() != 1 ||
        d.scalar_type() != (indices ? torch::kInt64 : torch::kFloat64)) {
      throw utils::runtime_error(
        ctx("contract"), std::string("Mode ") + letter[m] + ": expected 1-D " +
                           (indices ? "int64 indices" : "float64 weights"));
    }
    if (indices ? d.numel() > 0 && (d.min().item<int64_t>() < 0 ||
                                     d.max().item<int64_t>() >= size_[m])
                : d.numel() != size_[m]) {
      throw utils::runtime_error(ctx("contract"),
        std::string("Mode ") + letter[m] +
          (indices ? ": index out of range" : ": one weight per index"));
    }
  }
  if (joint && (!angle_matrix.defined() || angle_matrix.dim() != 2 ||
                 angle_matrix.size(0) != size_[MU] ||
                 angle_matrix.size(1) != size_[GA])) {
    throw utils::runtime_error(ctx("contract"),
      "Joint angle weights need an (N_mu, N_gamma) angle_matrix");
  }

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
      // Gather only the batched entries (advanced indexing) rather than
      // copying the whole core into batch-major order; the batch axis lands
      // at the first batched axis if those are adjacent, else in front
      std::vector<torch::indexing::TensorIndex> index(
        t.dim(), torch::indexing::Slice());
      std::vector<int64_t> batch_axes;
      for (int m : batch) {
        index[axis_of(m)] = select[m].data;
        batch_axes.push_back(axis_of(m));
      }
      std::sort(batch_axes.begin(), batch_axes.end());
      const bool adjacent = batch_axes.back() - batch_axes.front() + 1 ==
                            static_cast<int64_t>(batch_axes.size());
      t = t.index(index).movedim(adjacent ? batch_axes.front() : 0, 0);
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

torch::Tensor LayoutCross::moments_batched(const torch::Tensor& Yw) const
{
  const int64_t M = Yw.size(1), Nm = size_[MU], Ng = size_[GA];
  // Weights of the angle values leading each core: (M, a_k), or undefined
  std::vector<torch::Tensor> weight(K_);
  int64_t core_mu = -1, core_ga = -1;
  for (int64_t k = 0; k < K_; k++) {
    const auto& modes = core_modes_[k];
    for (size_t q = 0; q < modes.size(); q++) {
      if (modes[q] == MU)
        core_mu = k;
      if (modes[q] == GA)
        core_ga = k;
    }
  }
  const bool has_ga = layout_.contains(tensor::Dimension::AZIMUTHAL);
  auto Wt = Yw.t().contiguous(); // (M, Nm * Ng), polar major
  if (!has_ga) {
    if (core_modes_[core_mu][0] != MU)
      return {};
    weight[core_mu] = Wt;
  } else if (core_mu == core_ga) {
    const auto& modes = core_modes_[core_mu];
    if (modes.size() < 2)
      return {};
    if (modes[0] == MU && modes[1] == GA)
      weight[core_mu] = Wt;
    else if (modes[0] == GA && modes[1] == MU)
      weight[core_mu] =
        Wt.view({M, Nm, Ng}).transpose(1, 2).reshape({M, Nm * Ng});
    else
      return {};
  } else {
    if (core_modes_[core_mu][0] != MU || core_modes_[core_ga][0] != GA)
      return {};
    auto U = torch::zeros({M, Nm}, torch::kFloat64);
    auto V = torch::zeros({M, Ng}, torch::kFloat64);
    for (int64_t m = 0; m < M; m++) {
      auto W = Wt[m].view({Nm, Ng});
      auto [u, v] = outer_factors(W, 1e-10);
      if (!u.defined())
        return {}; // not separable: the general path reports it
      U[m].copy_(u);
      V[m].copy_(v);
    }
    weight[core_mu] = U;
    weight[core_ga] = V;
  }

  // Chain the cores with the moments as a batch: env (M, F, r), F the free
  // (space, energy) values met so far, in core order
  auto env = torch::ones({M, 1, 1}, torch::kFloat64);
  std::vector<int> free;
  for (int64_t k = 0; k < K_; k++) {
    const auto& c = cores_[k];
    const int64_t rl = c.size(0), nk = c.size(1), rr = c.size(2);
    const int64_t F = env.size(1);
    if (weight[k].defined()) {
      const int64_t a = weight[k].size(1);
      const int64_t R = (nk / a) * rr;
      // (M, a) x (rl, a, R) -> (rl, M, R): one product, no copy of the core
      auto Z = torch::matmul(weight[k], c.view({rl, a, R}));
      env = torch::einsum("mfa,amk->mfk", {env, Z});
    } else {
      env = torch::matmul(env, c.view({rl, nk * rr}));
    }
    env = env.reshape({M, -1, rr});
    for (int mode : core_modes_[k])
      if (mode == S || mode == E)
        free.push_back(mode);
  }
  // (M, free..., 1) -> (M, E, S)
  std::vector<int64_t> shape {M};
  for (int mode : free)
    shape.push_back(size_[mode]);
  env = env.reshape(shape);
  if (free.size() == 2 && free[0] == S)
    env = env.transpose(1, 2);
  return env.contiguous();
}

torch::Tensor LayoutCross::moments() const
{
  using Kind = Selector::Kind;
  auto Yw = problem_.harmonics.to(torch::kFloat64) *
            problem_.weights.to(torch::kFloat64).unsqueeze(1);
  // Every angle mode leading its core: all moments at once
  auto batched = moments_batched(Yw);
  if (batched.defined())
    return batched;
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
      "No TT held: run() first, or set keep_angular_flux to "
      "keep it after a solve");
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
