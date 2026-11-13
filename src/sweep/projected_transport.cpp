// ProjectedTransport: bordered Galerkin projections of the transport
// operator of one group set.
#include "ttnte/sweep/projected_transport.hpp"
#include "ttnte/linalg/dense_lu.hpp"
#include "ttnte/sweep/sweep_kernel.hpp"
#include "ttnte/utils/exception.hpp"
#include <ATen/Parallel.h>
#include <algorithm>

namespace ttnte::sweep {

using torch::indexing::Slice;

namespace {

/// Faces whose unit normals differ by less than this are flat (and two
/// flat faces share a normal).
constexpr double FLAT_TOL = 1e-12;

const auto F64 = torch::TensorOptions().dtype(torch::kFloat64);

/// @return (B, N) G x (transpose: G^T x) for pair values G.
torch::Tensor apply_pairs(const torch::Tensor& x, const torch::Tensor& row,
  const torch::Tensor& col, const torch::Tensor& values, bool transpose)
{
  const auto& out = transpose ? col : row;
  const auto& in = transpose ? row : col;
  auto y = torch::zeros_like(x);
  y.index_add_(1, out, x.index_select(1, in) * values);
  return y;
}

/// @return (p, G, Q) sum_a R(q, a) X[j, g, dofs[index(q, a)]], the basis
/// vectors at a face's quadrature points.
torch::Tensor at_points(const torch::Tensor& X_face, const torch::Tensor& index,
  const torch::Tensor& R)
{
  const int64_t p = X_face.size(0), ng = X_face.size(1);
  const int64_t Q = R.size(0), A = R.size(1);
  return (X_face.index_select(2, index.reshape({-1})).view({p, ng, Q, A}) * R)
    .sum(-1);
}

} // namespace

void ProjectedTransport::Growing::resize(int64_t num_rows, int64_t num_cols)
{
  const int64_t rcap = storage.defined() ? storage.size(0) : 0;
  const int64_t ccap = storage.defined() ? storage.size(1) : 0;
  if (num_rows > rcap || num_cols > ccap) {
    auto grown =
      torch::zeros({num_rows > rcap ? std::max(num_rows, 2 * rcap) : rcap,
                     num_cols > ccap ? std::max(num_cols, 2 * ccap) : ccap},
        F64);
    if (rows > 0 && cols > 0)
      grown.narrow(0, 0, rows).narrow(1, 0, cols).copy_(view());
    storage = grown;
  }
  rows = num_rows;
  cols = num_cols;
}

ProjectedTransport::ProjectedTransport(
  const SweepSolver& solver, size_t group_set, Options options)
  : solver_(solver), options_(options)
{
  const auto& S = solver_;
  if (S.comm_.size() > 1)
    throw utils::runtime_error(
      ctx("ProjectedTransport"), "Projected transport is serial in space");
  if (group_set >= S.group_indices_.size())
    throw utils::runtime_error(
      ctx("ProjectedTransport"), "Group set out of range");
  groups_ = S.group_indices_[group_set];
  num_groups_ = static_cast<int64_t>(groups_.size());
  num_space_ = S.fiber_length_;
  ndim_ = S.ndim_;
  num_moments_ = S.harmonics_.size(1);
  num_directions_ = S.omega_.size(0);
  const auto& sets = S.graph_->angle_sets();
  num_sets_ = static_cast<int64_t>(sets.size());
  streaming_.resize(ndim_);
  buffered_.resize(num_sets_);

  for (size_t u = 0; u < S.patches_.size(); u++) {
    const auto& P = S.patches_[u];
    for (size_t f = 0; f < P.links.size(); f++) {
      const auto& link = P.links[f];
      const auto& geom = P.disc->faces()[f];
      FaceProjection fp;
      fp.unit = u;
      fp.face = f;
      if (link.type == physics::BoundaryType::REFLECTIVE) {
        for (auto& list : buffered_)
          list.emplace_back(u, f);
      } else if (link.type == physics::BoundaryType::INTERNAL) {
        if (link.remote_rank >= 0)
          throw utils::runtime_error(
            ctx("ProjectedTransport"), "A neighbor is on another rank");
        fp.internal = true;
        fp.neighbor = static_cast<size_t>(link.neighbor);
        fp.neighbor_face = static_cast<size_t>(link.neighbor_face);
        fp.dependency.resize(num_sets_);
        for (int64_t a = 0; a < num_sets_; a++) {
          const EdgeState state = link.is_first
                                    ? sets[a].backward[link.interface]
                                    : sets[a].forward[link.interface];
          const bool lagged = state == EdgeState::LAGGED;
          fp.dependency[a] = state == EdgeState::DEPENDENCY ||
                             (lagged && options_.implicit_lagged);
          if (lagged && !options_.implicit_lagged)
            buffered_[a].emplace_back(u, f);
        }
      }

      // Flat: one unit normal at every quadrature point
      const auto norms = geom.normal.norm(2, 1);
      const auto unit_normals = geom.normal / norms.unsqueeze(1);
      const bool flat =
        options_.flat_faces && (norms > 0).all().template item<bool>() &&
        (unit_normals - unit_normals[0]).abs().max().template item<double>() <
          FLAT_TOL;
      if (flat) {
        const auto n = unit_normals[0].contiguous();
        for (size_t k = 0; k < normals_.size() && fp.normal < 0; k++)
          if ((normals_[k] - n).abs().max().template item<double>() < FLAT_TOL)
            fp.normal = static_cast<int64_t>(k);
        if (fp.normal < 0) {
          fp.normal = static_cast<int64_t>(normals_.size());
          normals_.push_back(n);
        }
        fp.measure = (norms * geom.weights).repeat({num_groups_}).contiguous();
      } else {
        curved_.push_back(faces_.size());
      }
      faces_.push_back(std::move(fp));
    }
  }
  outflow_.resize(normals_.size());
  inflow_.assign(num_sets_, std::vector<Growing>(normals_.size()));
  if (!curved_.empty())
    curved_cache_.resize(num_directions_);
  reset();
}

void ProjectedTransport::reset()
{
  rank_ = 0;
  basis_.resize(0, num_groups_ * num_space_);
  collision_.resize(0, 0);
  for (auto& s : streaming_)
    s.resize(0, 0);
  source_.resize(num_moments_, 0);
  for (auto& fp : faces_) {
    const int64_t rows =
      num_groups_ *
      solver_.patches_[fp.unit].disc->faces()[fp.face].weights.numel();
    fp.own.resize(rows, 0);
    if (fp.internal)
      fp.upwind.resize(rows, 0);
  }
  for (auto& F : outflow_)
    F.resize(0, 0);
  for (auto& set : inflow_)
    for (auto& K : set)
      K.resize(0, 0);
  for (auto& D : curved_cache_)
    D.resize(0, 0);
}

torch::Tensor ProjectedTransport::basis() const
{
  return basis_.view().view({rank_, num_groups_, num_space_});
}

// Bordering (Guo & Peng, eq. 3.11): with the basis grown from X (r0
// vectors) to [X P] (r1 = r0 + p), every projected operator B becomes
//   [X P]^T B [X P] = [ X^T B X (kept)   X^T B P ]
//                     [ P^T B X          P^T B P ]
// so only the new columns [X P]^T B P (r1 x p, the corner included) and,
// unless B is symmetric, the new rows P^T B [X P] (p x r1) are computed,
// and written into the Growing storage at J = [r0, r1). The old block is
// never touched, so the old basis vectors must never change (MGS appends).
void ProjectedTransport::extend(const torch::Tensor& X)
{
  if (X.dim() != 3 || X.size(1) != num_groups_ || X.size(2) != num_space_)
    throw utils::runtime_error(
      ctx("extend"), "X must be (p, |groups|, fiber_length)");
  const int64_t p = X.size(0);
  if (p == 0)
    return;
  const int64_t r0 = rank_, r1 = rank_ + p;
  const auto& S = solver_;
  const auto J = Slice(r0, r1);
  const auto group_index = torch::tensor(groups_, torch::kInt64);

  // Append P to the basis rows; `all` is [X P], `added` is P
  basis_.resize(r1, num_groups_ * num_space_);
  basis_.view().narrow(0, r0, p).copy_(X.reshape({p, -1}));
  const auto all =
    basis_.view().view({r1, num_groups_, num_space_}); // (r1, G, Ns)
  const auto added = all.narrow(0, r0, p);             // (p, G, Ns)

  // Volume terms. Mass and gradient matrices couple dofs of one unit only
  // (DG), and sigma_t is one value per (unit, group) (one fill per patch),
  // so each term is a sum over units of the unit's own blocks, applied
  // through its pair lists without assembling a global matrix. The sum over
  // the set's groups g is in the einsums: each basis vector spans all of
  // them (sum_g P_g^T B_g X_g).
  auto col_C = torch::zeros({r1, p}, F64);
  std::vector<torch::Tensor> col_S(ndim_), row_S(ndim_);
  for (int64_t i = 0; i < ndim_; i++) {
    col_S[i] = torch::zeros({r1, p}, F64);
    row_S[i] = torch::zeros({p, r1}, F64);
  }
  auto col_P = torch::zeros({num_moments_, p}, F64);
  for (size_t u = 0; u < S.patches_.size(); u++) {
    const auto& P = S.patches_[u];
    const auto& disc = *P.disc;
    const int64_t off = S.fiber_offsets_[u], N = disc.num_dofs();
    const auto Xu = all.narrow(2, off, N);                    // (r1, G, N)
    const auto Xn = added.narrow(2, off, N).reshape({-1, N}); // (p G, N)
    // Collision C = blockdiag_u(sigma_{t,u,g} M_u): symmetric, so the new
    // rows are the transpose of the new columns (written below)
    const auto sigma =
      P.sigma_t.index_select(0, group_index).view({1, num_groups_, 1});
    const auto MX = disc.apply_mass(Xn).view({p, num_groups_, N}) * sigma;
    col_C += torch::einsum("ign,jgn->ij", {Xu, MX});
    // Streaming S_i: G_i(a, b) = int R_b d_i R_a (the integrated-by-parts
    // volume term) is not symmetric, so the new columns X^T G_i P and the
    // new rows P^T G_i X = (G_i^T P)^T X both need an application
    for (int64_t i = 0; i < ndim_; i++) {
      const auto values = disc.grad_values()[i];
      const auto GX =
        apply_pairs(Xn, disc.pair_row(), disc.pair_col(), values, false)
          .view({p, num_groups_, N});
      const auto GtX =
        apply_pairs(Xn, disc.pair_row(), disc.pair_col(), values, true)
          .view({p, num_groups_, N});
      col_S[i] += torch::einsum("ign,jgn->ij", {Xu, GX});
      row_S[i] += torch::einsum("jgn,ign->ji", {GtX, Xu});
    }
    // Sources: X^T s_m per angular moment m (a vector, no rows to border);
    // P.source_moments is (M, all groups, N), taken at the set's groups.
    // Read now, so the basis belongs to this transport solve (reset())
    const auto s = P.source_moments.index_select(1, group_index);
    col_P += torch::einsum("mgn,jgn->mj", {s, added.narrow(2, off, N)});
  }
  // Write the borders: new columns [:, J], then new rows [J, :] (the
  // corner is written twice with the same value)
  collision_.resize(r1, r1);
  collision_.view().index_put_({Slice(), J}, col_C);
  collision_.view().index_put_({J, Slice()}, col_C.t());
  for (int64_t i = 0; i < ndim_; i++) {
    streaming_[i].resize(r1, r1);
    streaming_[i].view().index_put_({Slice(), J}, col_S[i]);
    streaming_[i].view().index_put_({J, Slice()}, row_S[i]);
  }
  source_.resize(num_moments_, r1);
  source_.view().index_put_({Slice(), J}, col_P);

  // Faces, first the traces: the new vectors' values at each face's
  // quadrature points, (T_f P)_{g,q} = sum_a R_a(q) P_g[dof(a)], appended
  // as new columns of fp.own (rows g * Qf + q). Every face integral below
  // is then a dense product over quadrature points. Internal faces also
  // keep the upwind neighbor's trace at the same points, (T_f^nb P), in
  // fp.upwind: the inflow a DEPENDENCY edge brings in from the neighbor.
  for (auto& fp : faces_) {
    const auto& P = S.patches_[fp.unit];
    const auto& geom = P.disc->faces()[fp.face];
    const int64_t off = S.fiber_offsets_[fp.unit];
    const auto own = at_points(
      added.narrow(2, off, P.disc->num_dofs()).index_select(2, geom.dofs),
      geom.basis_index, geom.basis_values);
    fp.own.resize(fp.own.rows, r1);
    fp.own.view().index_put_(
      {Slice(), J}, own.permute({1, 2, 0}).reshape({-1, p}));
    if (!fp.internal)
      continue;
    const auto& link = P.links[fp.face];
    const auto& nb = *S.patches_[fp.neighbor].disc;
    const auto X_nb =
      added.narrow(2, S.fiber_offsets_[fp.neighbor], nb.num_dofs())
        .index_select(2, nb.faces()[fp.neighbor_face].dofs);
    // Conforming: the neighbor's face dofs mapped onto this face's basis;
    // non-conforming: the neighbor's own basis at the shared points
    const auto up =
      link.coupling_basis.defined()
        ? at_points(X_nb, link.coupling_index, link.coupling_basis)
        : at_points(X_nb,
            link.trace_map.index_select(0, geom.basis_index.reshape({-1}))
              .view_as(geom.basis_index),
            geom.basis_values);
    fp.upwind.resize(fp.upwind.rows, r1);
    fp.upwind.view().index_put_(
      {Slice(), J}, up.permute({1, 2, 0}).reshape({-1, p}));
  }

  // Flat faces (constant unit normal n): (Omega . n_q)_+- w_q =
  // (Omega . n)_+- |n_q| w_q factors, so the direction only scales fixed
  // matrices. Summed over the faces sharing a normal (linear, so the
  // borders add):
  //   F_n     = sum_f (T_f X)^T W_f (T_f X)      outflow, symmetric
  //   K_{a,n} = sum_f (T_f X)^T W_f (T_f^nb X)   inflow over the faces
  //             angle set a sweeps upwind first (DEPENDENCY), test on the
  //             own side and trial on the neighbor's: not symmetric
  // with W_f = diag(|n_q| w_q) (fp.measure, repeated per group).
  // reduced_system() weights them by (Omega . n)_+ and (Omega . n)_-.
  const int64_t num_normals = static_cast<int64_t>(normals_.size());
  std::vector<torch::Tensor> col_F(num_normals);
  std::vector<std::vector<torch::Tensor>> col_K(num_sets_), row_K(num_sets_);
  for (int64_t n = 0; n < num_normals; n++)
    col_F[n] = torch::zeros({r1, p}, F64);
  for (int64_t a = 0; a < num_sets_; a++) {
    col_K[a].assign(num_normals, torch::Tensor());
    row_K[a].assign(num_normals, torch::Tensor());
    for (int64_t n = 0; n < num_normals; n++) {
      col_K[a][n] = torch::zeros({r1, p}, F64);
      row_K[a][n] = torch::zeros({p, r1}, F64);
    }
  }
  for (const auto& fp : faces_) {
    if (fp.normal < 0)
      continue;
    // F's new columns: [X P]^T W (T P)
    const auto PX = fp.own.view();
    const auto wPXn = PX.index({Slice(), J}) * fp.measure.unsqueeze(1);
    col_F[fp.normal] += PX.t().mm(wPXn);
    if (!fp.internal)
      continue;
    // K's new columns [X P]^T W (T^nb P) and new rows (T P)^T W [T^nb X,
    // T^nb P], added to every angle set that has this edge as a dependency
    // (lagged sets take it from the buffers, buffered_rhs())
    const auto NB = fp.upwind.view();
    const auto col =
      PX.t().mm(NB.index({Slice(), J}) * fp.measure.unsqueeze(1));
    const auto row = wPXn.t().mm(NB);
    for (int64_t a = 0; a < num_sets_; a++) {
      if (!fp.dependency[a])
        continue;
      col_K[a][fp.normal] += col;
      row_K[a][fp.normal] += row;
    }
  }
  for (int64_t n = 0; n < num_normals; n++) {
    outflow_[n].resize(r1, r1);
    outflow_[n].view().index_put_({Slice(), J}, col_F[n]);
    outflow_[n].view().index_put_({J, Slice()}, col_F[n].t());
    for (int64_t a = 0; a < num_sets_; a++) {
      inflow_[a][n].resize(r1, r1);
      inflow_[a][n].view().index_put_({Slice(), J}, col_K[a][n]);
      inflow_[a][n].view().index_put_({J, Slice()}, row_K[a][n]);
    }
  }
  rank_ = r1;
}

// Curved faces: n_q / |n_q| varies along the face, so (Omega . n_q)_+-
// does not factor out (it can even change sign partway along a face) and
// the face integrals are not a fixed matrix times a scalar in Omega. Each
// direction keeps its own projection
//   D_d = sum_f (T_f X)^T W^+_{d,f} (T_f X)
//       - sum_{f DEPENDENCY for set(d)} (T_f X)^T W^-_{d,f} (T_f^nb X),
//   W^+-_{d,f} = diag((Omega_d . n_q)_+- w_q),
// with n_q unnormalized (|n_q| is the face Jacobian), the kernel's upwind
// weights. It is bordered lazily: extend() only grows the traces, and the
// cache, bordered up to its own size r0, catches up to the current rank r1
// here, when d is next assembled (several extensions at once if d was
// skipped meanwhile; J = [r0, r1) is then wider). Only D_d is written, so
// directions can be assembled concurrently.
void ProjectedTransport::update_curved(int64_t d)
{
  auto& D = curved_cache_[d];
  const int64_t r0 = D.rows, r1 = rank_;
  if (r0 == r1)
    return; // up to date
  const auto& S = solver_;
  const int64_t a = S.graph_->direction_set().template data_ptr<int64_t>()[d];
  const auto omega = S.omega_[d];
  const auto J = Slice(r0, r1);
  auto col = torch::zeros({r1, r1 - r0}, F64);
  auto row = torch::zeros({r1 - r0, r1}, F64);
  for (size_t k : curved_) {
    const auto& fp = faces_[k];
    const auto& geom = S.patches_[fp.unit].disc->faces()[fp.face];
    // c_q = (Omega . n_q)_+- w_q, as the kernel's upwind weights, repeated
    // per group to match the trace rows g * Qf + q
    const auto dot = geom.normal.mv(omega);
    const auto out = (dot.clamp_min(0.0) * geom.weights).repeat({num_groups_});
    // Outflow (symmetric): new columns [X P]^T W+ (T P); new rows their
    // transpose, accumulated as rows since `row` also carries the inflow
    const auto PX = fp.own.view();
    const auto wPXn = PX.index({Slice(), J}) * out.unsqueeze(1);
    col += PX.t().mm(wPXn);
    row += wPXn.t().mm(PX);
    if (!fp.internal || !fp.dependency[a])
      continue;
    // Dependency inflow (not symmetric, subtracted): new columns
    // [X P]^T W- (T^nb P) and new rows (T P)^T W- [T^nb X, T^nb P]
    const auto in =
      ((-dot).clamp_min(0.0) * geom.weights).repeat({num_groups_});
    const auto NB = fp.upwind.view();
    col -= PX.t().mm(NB.index({Slice(), J}) * in.unsqueeze(1));
    row -= (PX.index({Slice(), J}) * in.unsqueeze(1)).t().mm(NB);
  }
  D.resize(r1, r1);
  D.view().index_put_({Slice(), J}, col);
  D.view().index_put_({J, Slice()}, row);
}

// The inflow that is data, not unknown, for direction d: lagged internal
// edges (cycle breaks of d's angle set) and reflective faces, whose traces
// sit in the FaceInflow buffers (trace_previous, the reflected traces).
// Rather than re-deriving the trace layouts (reflection maps, non-conforming
// coupling bases, lagged neighbor dofs), each face's inflow is assembled by
// the sweep kernel's own buffered_inflow(), exactly as a sweep adds it to
// the unit's right-hand side b, and then projected: rhs += X_face^T b_face.
// Only the face's dofs of b are touched, so b is read and re-zeroed there.
// TODO: make_unit per face and direction; cache it if timings show it.
torch::Tensor ProjectedTransport::buffered_rhs(int64_t d) const
{
  if (d < 0 || d >= num_directions_)
    throw utils::runtime_error(ctx("buffered_rhs"), "Direction out of range");
  const auto& S = solver_;
  const int64_t a = S.graph_->direction_set().template data_ptr<int64_t>()[d];
  auto rhs = torch::zeros({rank_}, F64);
  if (buffered_[a].empty() || rank_ == 0)
    return rhs;
  // A one-direction block over the set's groups, and per face the unit's
  // sweep view in d's angle set: what the kernel reads the buffers through
  const auto& set = S.graph_->angle_sets()[a];
  const std::vector<int64_t> dirs {d};
  const auto block = S.make_block(dirs, groups_);
  const auto X = basis();
  std::vector<double> c, b;
  for (const auto& [u, f] : buffered_[a]) {
    const auto& P = S.patches_[u];
    const auto unit = S.make_unit(set, static_cast<int64_t>(u));
    const auto& geom = P.disc->faces()[f];
    const int64_t N = P.disc->num_dofs();
    const int64_t Nf = geom.num_dofs();
    const int64_t* dofs = geom.dofs.template data_ptr<int64_t>();
    c.resize(P.disc->view().max_face_points);
    b.assign(N, 0.0);
    auto load = torch::empty({Nf}, F64);
    double* l = load.template data_ptr<double>();
    // The basis on the face's dofs, (r, G, Nf)
    const auto X_face =
      X.narrow(2, S.fiber_offsets_[u], N).index_select(2, geom.dofs);
    for (int64_t bg = 0; bg < num_groups_; bg++) {
      // Inflow or not depends on d only, so a face that is outflow for d is
      // skipped for every group
      if (!buffered_inflow(unit, block, d, groups_[bg], f, c.data(), b.data()))
        break;
      // Gather the face's entries of b (the only ones written) and clear
      // them for the next group
      for (int64_t k = 0; k < Nf; k++) {
        l[k] = b[dofs[k]];
        b[dofs[k]] = 0.0;
      }
      // X_g^T b_g on the face's dofs
      rhs += X_face.select(1, bg).mv(load);
    }
  }
  return rhs;
}

std::pair<torch::Tensor, torch::Tensor> ProjectedTransport::reduced_system(
  int64_t d)
{
  if (d < 0 || d >= num_directions_)
    throw utils::runtime_error(ctx("reduced_system"), "Direction out of range");
  const auto& S = solver_;
  const int64_t a = S.graph_->direction_set().template data_ptr<int64_t>()[d];
  const auto omega = S.omega_[d];
  const double* w = omega.template data_ptr<double>();

  auto A = collision_.view().clone();
  for (int64_t i = 0; i < ndim_; i++)
    A -= w[i] * streaming_[i].view();
  for (size_t n = 0; n < normals_.size(); n++) {
    const double dot = normals_[n].dot(omega).template item<double>();
    if (dot > 0.0)
      A += dot * outflow_[n].view();
    else if (dot < 0.0)
      A += dot * inflow_[a][n].view(); // -(Omega . n)_- K
  }
  if (!curved_.empty()) {
    update_curved(d);
    A += curved_cache_[d].view();
  }
  auto b = S.harmonics_[d].matmul(source_.view()) + buffered_rhs(d);
  return {A, b};
}

torch::Tensor ProjectedTransport::solve(const std::vector<int64_t>& directions)
{
  const int64_t D = static_cast<int64_t>(directions.size());
  const int64_t r = rank_;
  // Each direction borders only its own curved cache, so the directions of
  // one call must be distinct to be assembled concurrently
  std::vector<uint8_t> seen(static_cast<size_t>(num_directions_), 0);
  for (int64_t d : directions) {
    if (d < 0 || d >= num_directions_)
      throw utils::runtime_error(ctx("solve"), "Direction out of range");
    if (seen[d]++)
      throw utils::runtime_error(ctx("solve"), "Directions must be distinct");
  }
  auto C = torch::zeros({D, r}, F64);
  if (D == 0 || r == 0)
    return C;
  double* c = C.data_ptr<double>();
  // Per direction: assemble (r x r, row-major), LU in place, solve
  std::vector<uint8_t> singular(static_cast<size_t>(D), 0);
  at::parallel_for(0, D, 1, [&](int64_t i0, int64_t i1) {
    std::vector<int64_t> ipiv(static_cast<size_t>(r));
    for (int64_t i = i0; i < i1; i++) {
      auto [A, b] = reduced_system(directions[i]);
      A = A.contiguous();
      double* a = A.data_ptr<double>();
      if (linalg::dense::lu_factor(a, r, ipiv.data()) != 0) {
        singular[i] = 1;
        continue;
      }
      double* x = c + i * r;
      std::copy_n(b.contiguous().data_ptr<double>(), r, x);
      linalg::dense::lu_solve(a, r, ipiv.data(), x);
    }
  });
  for (int64_t i = 0; i < D; i++)
    if (singular[i])
      throw utils::runtime_error(ctx("solve"),
        "The reduced matrix of direction " + std::to_string(directions[i]) +
          " is singular (rank-deficient basis?)");
  return C;
}

torch::Tensor ProjectedTransport::residuals(
  const std::vector<int64_t>& directions, const torch::Tensor& C) const
{
  const int64_t D = static_cast<int64_t>(directions.size());
  const int64_t r = rank_;
  if (C.dim() != 2 || C.size(0) != D || C.size(1) != r)
    throw utils::runtime_error(
      ctx("residuals"), "C must be (len(directions), rank)");
  for (int64_t d : directions)
    if (d < 0 || d >= num_directions_)
      throw utils::runtime_error(ctx("residuals"), "Direction out of range");
  auto R = torch::empty({D, num_groups_, num_space_}, F64);
  if (D == 0)
    return R;
  const auto coef = C.to(F64).contiguous();
  const auto X = basis().reshape({r, -1}); // (r, G Ns)
  double* out = R.data_ptr<double>();
  const int64_t stride = num_groups_ * num_space_;
  at::parallel_for(0, D, 1, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; i++) {
      // x = sum_j c_j x_j, then r = b_d - A_d x through the sweep kernel. A
      // zero row (the right-hand side b_d = r(0)) skips the O(r G Ns) product
      const double* ci = coef.data_ptr<double>() + i * r;
      const bool zero =
        std::all_of(ci, ci + r, [](double v) { return v == 0.0; });
      const auto x =
        zero ? torch::zeros({stride}, F64) : coef[i].matmul(X).contiguous();
      solver_.transport_residual(
        directions[i], groups_, x.data_ptr<double>(), out + i * stride);
    }
  });
  return R;
}

} // namespace ttnte::sweep
