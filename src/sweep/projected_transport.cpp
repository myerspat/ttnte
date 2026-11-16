// ProjectedTransport: bordered Galerkin projections of the transport
// operator of one group set.
//
// What it computes. For a basis X of the fiber space (r vectors, each the
// angular flux of one direction over the set's groups and every local dof)
// and any direction d, the reduced system
//   A_d = X^T L_d X,   b_d = X^T (source_d + buffered inflow_d),
// with L_d the sweep's own operator for d (sigma_t M - Omega_d . G + upwind
// face terms), and the full-space residuals b_d - L_d X c_d of reduced
// solutions. GreedyGalerkin grows X one swept direction at a time, so every
// projection is bordered (extend()) rather than recomputed.
//
// How L_d splits, so that only cheap per-direction work is left:
//   - volume: sigma M (collision) and G_i (streaming), independent of d;
//     A_d gets C - sum_i Omega_i S_i with C = X^T sigma M X, S_i = X^T G_i X;
//   - flat faces (one unit normal n): the upwind weights factor as (Omega .
//     n)_+- times a fixed per-point measure, so per distinct normal one
//     outflow matrix F_n and per angle set one inflow matrix K_{a,n};
//   - curved faces: per sign class of Omega . n_q (FaceClasses, linear in
//     Omega within a class), or per direction (curved_cache_, bordered
//     lazily in update_curved());
//   - sources: X^T s_m per angular moment, b_d = Y(d)^T [X^T s_m];
//   - buffered inflow (lagged edges, reflective faces): data, assembled per
//     direction by the kernel's buffered_inflow() and projected.
//
// Sampling units. A unit is a direction (its flux over the set's groups,
// one reduced system per direction: Guo & Peng's multigroup), or with
// Options::spatial_basis a (direction, group) pair u = d G + g (pair
// sampling, our ablation): basis vectors are then spatial (basis_groups_ =
// 1), the collision, sources and buffered inflow are per group, and the
// streaming and face projections are shared by every group.
//
// Layouts. A fiber is (num_groups, num_space) flattened as g * Ns + dof, the
// units' dofs end to end at the solver's fiber_offsets_. The basis is one
// row per vector (basis_, (r, G Ns)). Face traces (T X: the basis at the
// face quadrature points) are one global matrix with one row per (face,
// group, point), face f's rows row0 + g * Qf + q; flat faces come first,
// grouped by normal, then curved ones (build_traces()).
//
// Notation used below: X the current basis (r0 vectors), P the vectors being
// added (p of them), [X P] the grown basis (r1 = r0 + p), J = [r0, r1) the
// new columns; T / T^nb the own / upwind neighbor trace at a face's points;
// W a diagonal of upwind weights; D the number of directions.
//
// Lifecycle: construct (once per solver setup), then per transport solve
// reset(), extend() as vectors arrive, and reduced_system() / solve() /
// residuals() in between. The sources and lagged traces are read during
// that solve, so a basis belongs to one transport solve.
#include "ttnte/sweep/projected_transport.hpp"
#include "ttnte/linalg/dense_lu.hpp"
#include "ttnte/linalg/lapack.hpp"
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

} // namespace

// =================================================================
// Growing storage
// =================================================================

// Resize the logical (rows, cols) view; the storage grows by doubling and
// keeps the old block, so bordering never copies more than amortized O(1)
// per element. New entries are zero.
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

// =================================================================
// Construction: classify faces, build the global volume and trace data
// =================================================================

// The constructor, in order:
//   1. classify every face of every unit: reflective (inflow buffered),
//      internal (its neighbor, and per angle set whether the neighbor is
//      swept first: DEPENDENCY, an unknown; or LAGGED, buffered data),
//      flat (its distinct unit normal) or curved;
//   2. build_volume(), build_traces(): the global pair lists and traces;
//   3. curved faces: by sign class where that pays (max_face_classes,
//      max_class_ratio), else per direction;
//   4. reset(): an empty basis.
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
  // Pair sampling: one group per basis vector, one unit per (d, g)
  basis_groups_ = options_.spatial_basis ? 1 : num_groups_;
  num_units_ =
    options_.spatial_basis ? num_directions_ * num_groups_ : num_directions_;
  const auto& sets = S.graph_->angle_sets();
  num_sets_ = static_cast<int64_t>(sets.size());
  collision_.resize(options_.spatial_basis ? num_groups_ : 1);
  streaming_.resize(ndim_);
  buffered_.resize(num_sets_);

  // 1. Every face of every unit
  for (size_t u = 0; u < S.patches_.size(); u++) {
    const auto& P = S.patches_[u];
    for (size_t f = 0; f < P.links.size(); f++) {
      const auto& link = P.links[f];
      const auto& geom = P.disc->faces()[f];
      FaceProjection fp;
      fp.unit = u;
      fp.face = f;
      // Reflective: the reflected trace is data for every angle set.
      // Internal: per angle set, an upwind neighbor swept first in that set
      // (DEPENDENCY) puts its trace in A_d; one swept after it (LAGGED, a
      // cycle break) leaves it in the buffers, unless implicit_lagged.
      // Vacuum faces need nothing (no inflow)
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

      // Flat: one unit normal at every quadrature point (to FLAT_TOL); flat
      // faces sharing a normal share F_n and K_{a,n}. fp.measure is |n_q| w_q
      // repeated per group, matching the trace rows g * Qf + q
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
        fp.measure =
          (norms * geom.weights).repeat({basis_groups_}).contiguous();
      } else {
        curved_.push_back(faces_.size());
      }
      faces_.push_back(std::move(fp));
    }
  }
  outflow_.resize(normals_.size());
  inflow_.assign(num_sets_, std::vector<Growing>(normals_.size()));
  // 2. Global volume pair lists and face traces
  build_volume();
  build_traces();

  // 3. Curved faces by sign class: the distinct sign vectors sign(Omega_d .
  // n_q) over the face's points, and each direction's class. A face with
  // more classes than max_face_classes keeps the per-direction projection
  for (size_t k : curved_) {
    const auto& fp = faces_[k];
    const auto& geom = S.patches_[fp.unit].disc->faces()[fp.face];
    if (!options_.sign_split) {
      per_direction_.push_back(k);
      continue;
    }
    const auto dot = S.omega_.to(F64).mm(geom.normal.to(F64).t()); // (D, Qf)
    const auto sign = ((dot > 0).to(torch::kInt8) - (dot < 0).to(torch::kInt8));
    const auto [unique, inverse, counts] =
      at::unique_dim(sign, 0, /*sorted=*/true, /*return_inverse=*/true);
    const int64_t C = unique.size(0);
    if (C > options_.max_face_classes) {
      per_direction_.push_back(k);
      continue;
    }
    FaceClasses fc;
    fc.face = k;
    fc.num_classes = C;
    const auto inv = inverse.to(torch::kInt64).contiguous();
    fc.class_of.assign(
      inv.data_ptr<int64_t>(), inv.data_ptr<int64_t>() + num_directions_);
    // Per point (the traces' rows g * Qf + q are summed over g first)
    fc.positive = (unique > 0).to(F64).contiguous();
    fc.negative = (unique < 0).to(F64).contiguous();
    fc.moments = (geom.normal.to(F64).t() * geom.weights.to(F64).unsqueeze(0))
                   .contiguous();
    classes_.push_back(std::move(fc));
  }
  // Too many class matrices against the per-direction caches they replace:
  // every curved face per direction (Options::max_class_ratio)
  int64_t num_class_matrices = 0;
  for (const auto& fc : classes_)
    num_class_matrices +=
      fc.num_classes * ndim_ * (faces_[fc.face].internal ? 2 : 1);
  if (static_cast<double>(num_class_matrices) >
      options_.max_class_ratio * static_cast<double>(num_directions_)) {
    for (const auto& fc : classes_)
      per_direction_.push_back(fc.face);
    classes_.clear();
  }
  if (!per_direction_.empty())
    curved_cache_.resize(num_directions_);
  // 4. An empty basis
  reset();
}

// The volume terms of every unit as one pair list over the fiber: mass and
// gradient pairs couple dofs of one unit only (DG), so offsetting each
// unit's pairs into the fiber gives the block-diagonal operators, applied
// in one pass. sigma_t is one value per (unit, group) (one fill per patch).
void ProjectedTransport::build_volume()
{
  const auto& S = solver_;
  const auto group_index = torch::tensor(groups_, torch::kInt64);
  std::vector<torch::Tensor> rows, cols, mass, grad;
  sigma_ = torch::zeros({num_groups_, num_space_}, F64);
  for (size_t u = 0; u < S.patches_.size(); u++) {
    const auto& P = S.patches_[u];
    const auto& disc = *P.disc;
    const int64_t off = S.fiber_offsets_[u], N = disc.num_dofs();
    rows.push_back(disc.pair_row() + off);
    cols.push_back(disc.pair_col() + off);
    mass.push_back(disc.mass_values().to(F64));
    grad.push_back(disc.grad_values().to(F64).reshape({ndim_, -1}));
    sigma_.narrow(1, off, N).copy_(P.sigma_t.to(F64)
        .index_select(0, group_index)
        .unsqueeze(1)
        .expand({num_groups_, N}));
  }
  volume_row_ = torch::cat(rows).contiguous();
  volume_col_ = torch::cat(cols).contiguous();
  volume_mass_ = torch::cat(mass).contiguous();
  volume_grad_ = torch::cat(grad, 1).contiguous();
}

// The traces of every face in one matrix: flat faces first, grouped by
// normal (so that each normal's faces are one block of rows), then curved
// ones. Per row g * Qf + q of a face, the entries of the flattened (G Ns)
// basis vector and the basis values that form (R X)_{g,q}, and for an
// INTERNAL face the neighbor's at the same points (conforming: its face
// dofs mapped onto this face's basis; non-conforming: its own basis at the
// shared points), zero-padded to the widest face.
void ProjectedTransport::build_traces()
{
  const auto& S = solver_;
  // Rows per basis group (one with spatial_basis): a trace row is a point of
  // one group of a basis vector, read at g Ns + dof
  const int64_t G = basis_groups_, Ns = num_space_;
  // Row ranges: assign each face its block [row0, row0 + G Qf) in the order
  // flat-by-normal, then curved; `order` is that face order
  std::vector<size_t> order;
  normal_rows_.assign(normals_.size(), {0, 0});
  int64_t row = 0;
  auto place = [&](size_t k) {
    auto& fp = faces_[k];
    const auto& geom = S.patches_[fp.unit].disc->faces()[fp.face];
    fp.row0 = row;
    fp.rows = G * geom.weights.numel();
    row += fp.rows;
    order.push_back(k);
  };
  for (size_t n = 0; n < normals_.size(); n++) {
    normal_rows_[n].first = row;
    for (size_t k = 0; k < faces_.size(); k++)
      if (faces_[k].normal == static_cast<int64_t>(n))
        place(k);
    normal_rows_[n].second = row - normal_rows_[n].first;
  }
  for (size_t k : curved_)
    place(k);
  num_trace_rows_ = row;

  // Gather data per row: a trace value is sum_a value[row, a] * x[index[row,
  // a]], x a flattened (G Ns) fiber, so the index already includes g Ns.
  // Per group g the flattened entries g Ns + dof, (G Qf, A)
  const auto group_offsets =
    (torch::arange(G, torch::kInt64) * Ns).view({G, 1, 1});
  auto per_group = [&](const torch::Tensor& dofs) {
    return (dofs.unsqueeze(0) + group_offsets).reshape({-1, dofs.size(1)});
  };
  auto per_group_values = [&](const torch::Tensor& values) {
    return values.to(F64)
      .unsqueeze(0)
      .expand({G, -1, -1})
      .reshape({-1, values.size(1)});
  };
  std::vector<torch::Tensor> own_i, own_v, up_i, up_v, normals, weights;
  std::vector<std::vector<torch::Tensor>> dependency(num_sets_);
  int64_t width = 1;
  for (size_t k : order) {
    const auto& fp = faces_[k];
    const auto& P = S.patches_[fp.unit];
    const auto& geom = P.disc->faces()[fp.face];
    const int64_t off = S.fiber_offsets_[fp.unit];
    const int64_t Q = geom.weights.numel();
    // Per row g * Qf + q: the point's normal and weight, and per set
    // whether its inflow is an unknown (DEPENDENCY)
    normals.push_back(geom.normal.to(F64)
        .unsqueeze(0)
        .expand({G, Q, ndim_})
        .reshape({G * Q, ndim_}));
    weights.push_back(geom.weights.to(F64).repeat({G}));
    for (int64_t a = 0; a < num_sets_; a++)
      dependency[a].push_back(
        torch::full({G * Q}, fp.internal && fp.dependency[a] ? 1.0 : 0.0, F64));
    const auto own = geom.dofs.index_select(0, geom.basis_index.reshape({-1}))
                       .view_as(geom.basis_index) +
                     off;
    own_i.push_back(per_group(own));
    own_v.push_back(per_group_values(geom.basis_values));
    width = std::max(width, own.size(1));
    if (!fp.internal) {
      up_i.push_back(torch::zeros({G * Q, 1}, torch::kInt64));
      up_v.push_back(torch::zeros({G * Q, 1}, F64));
      continue;
    }
    any_internal_ = true;
    const auto& link = P.links[fp.face];
    const auto& nb = *S.patches_[fp.neighbor].disc;
    const auto& nb_dofs = nb.faces()[fp.neighbor_face].dofs;
    const int64_t nb_off = S.fiber_offsets_[fp.neighbor];
    torch::Tensor index, values;
    if (link.coupling_basis.defined()) {
      index = link.coupling_index;
      values = link.coupling_basis;
    } else {
      index = link.trace_map.index_select(0, geom.basis_index.reshape({-1}))
                .view_as(geom.basis_index);
      values = geom.basis_values;
    }
    if (index.size(0) != Q)
      throw utils::runtime_error(ctx("build_traces"),
        "Neighbor trace points do not match the face's quadrature");
    const auto up =
      nb_dofs.index_select(0, index.reshape({-1})).view_as(index) + nb_off;
    up_i.push_back(per_group(up));
    up_v.push_back(per_group_values(values));
    width = std::max(width, up.size(1));
  }
  // Concatenate the faces' rows, zero-padding every face to the widest A
  // (a padded entry has value 0 and index 0: it adds nothing)
  auto stack = [&](std::vector<torch::Tensor>& parts, bool is_index) {
    for (auto& t : parts)
      if (t.size(1) < width) {
        auto padded = torch::zeros({t.size(0), width},
          is_index ? torch::TensorOptions().dtype(torch::kInt64) : F64);
        padded.narrow(1, 0, t.size(1)).copy_(t);
        t = padded;
      }
    return parts.empty() ? (is_index ? torch::zeros({0, width}, torch::kInt64)
                                     : torch::zeros({0, width}, F64))
                         : torch::cat(parts).contiguous();
  };
  row_normal_ = normals.empty() ? torch::zeros({0, ndim_}, F64)
                                : torch::cat(normals).contiguous();
  row_weight_ =
    weights.empty() ? torch::zeros({0}, F64) : torch::cat(weights).contiguous();
  std::vector<torch::Tensor> dep_rows;
  for (int64_t a = 0; a < num_sets_; a++)
    dep_rows.push_back(dependency[a].empty() ? torch::zeros({0}, F64)
                                             : torch::cat(dependency[a]));
  row_dependency_ = torch::stack(dep_rows).contiguous();
  own_index_ = stack(own_i, true).reshape({-1});
  own_value_ = stack(own_v, false);
  upwind_index_ = stack(up_i, true).reshape({-1});
  upwind_value_ = stack(up_v, false);

  // Per normal its weights, and per angle set those of its DEPENDENCY faces
  normal_measure_.assign(normals_.size(), torch::Tensor());
  dependency_measure_.assign(
    num_sets_, std::vector<torch::Tensor>(normals_.size()));
  for (size_t n = 0; n < normals_.size(); n++) {
    std::vector<torch::Tensor> w;
    std::vector<std::vector<torch::Tensor>> dep(num_sets_);
    std::vector<uint8_t> any(num_sets_, 0);
    for (size_t k : order) {
      const auto& fp = faces_[k];
      if (fp.normal != static_cast<int64_t>(n))
        continue;
      w.push_back(fp.measure);
      for (int64_t a = 0; a < num_sets_; a++) {
        const bool on = fp.internal && fp.dependency[a];
        any[a] |= on;
        dep[a].push_back(on ? fp.measure : torch::zeros_like(fp.measure));
      }
    }
    if (w.empty())
      continue;
    normal_measure_[n] = torch::cat(w).contiguous();
    for (int64_t a = 0; a < num_sets_; a++)
      if (any[a])
        dependency_measure_[a][n] = torch::cat(dep[a]).contiguous();
  }
}

// The set's sources over the whole fiber, (M, G, Ns), gathered once per
// transport solve (reset() drops them): the solver rewrites the units'
// source moments between solves, never during one.
void ProjectedTransport::gather_sources() const
{
  if (source_all_.defined())
    return;
  const auto& S = solver_;
  const auto group_index = torch::tensor(groups_, torch::kInt64);
  source_all_ = torch::zeros({num_moments_, num_groups_, num_space_}, F64);
  for (size_t u = 0; u < S.patches_.size(); u++) {
    const auto& P = S.patches_[u];
    source_all_.narrow(2, S.fiber_offsets_[u], P.disc->num_dofs())
      .copy_(P.source_moments.index_select(1, group_index));
  }
}

// Unit u's buffered inflow as a full vector (added to `out`, (basis_groups,
// Ns)): the part of b_u that residuals() needs unprojected. Same assembly as
// buffered_rhs(), without the projection onto X.
void ProjectedTransport::add_buffered_inflow(int64_t u_index, double* out) const
{
  const auto& S = solver_;
  const int64_t d = unit_direction(u_index), g_unit = unit_group(u_index);
  const int64_t a = S.graph_->direction_set().template data_ptr<int64_t>()[d];
  if (buffered_[a].empty())
    return;
  const auto& set = S.graph_->angle_sets()[a];
  const std::vector<int64_t> dirs {d};
  const auto block = S.make_block(dirs, groups_);
  std::vector<double> c, b;
  for (const auto& [u, f] : buffered_[a]) {
    const auto& P = S.patches_[u];
    const auto unit = S.make_unit(set, static_cast<int64_t>(u));
    const auto& geom = P.disc->faces()[f];
    const int64_t N = P.disc->num_dofs();
    const int64_t Nf = geom.num_dofs();
    const int64_t* dofs = geom.dofs.template data_ptr<int64_t>();
    const int64_t off = S.fiber_offsets_[u];
    c.resize(P.disc->view().max_face_points);
    b.assign(N, 0.0);
    // The unit's groups: all of the set's (written at bg Ns), or its own
    // (spatial_basis, written at 0)
    const int64_t g0 = g_unit < 0 ? 0 : g_unit;
    const int64_t g1 = g_unit < 0 ? num_groups_ : g_unit + 1;
    for (int64_t bg = g0; bg < g1; bg++) {
      // A face that is outflow for d is skipped for every group
      if (!buffered_inflow(unit, block, d, groups_[bg], f, c.data(), b.data()))
        break;
      // Only the face's entries of b are written: add them, clear them
      double* o = out + (g_unit < 0 ? bg : 0) * num_space_ + off;
      for (int64_t k = 0; k < Nf; k++) {
        o[dofs[k]] += b[dofs[k]];
        b[dofs[k]] = 0.0;
      }
    }
  }
}

// =================================================================
// Curved faces by sign class
// =================================================================

// Per curved face its number of sign classes, -1 if projected per direction
// (testing aid).
std::vector<int64_t> ProjectedTransport::face_classes() const
{
  std::vector<int64_t> out;
  for (size_t k : curved_) {
    int64_t n = -1;
    for (const auto& fc : classes_)
      if (fc.face == k)
        n = fc.num_classes;
    out.push_back(n);
  }
  return out;
}

// Border the class matrices O_{c,i} (and I_{c,i} on internal faces) with
// the new columns J = [r0, r1), from the traces extend() just wrote.
void ProjectedTransport::extend_classes(int64_t r0, int64_t r1)
{
  if (classes_.empty())
    return;
  const int64_t p = r1 - r0;
  // Capacity doubles; the old blocks are kept (they are never rewritten)
  if (r1 > classes_capacity_) {
    int64_t cap = std::max<int64_t>(4, classes_capacity_);
    while (cap < r1)
      cap *= 2;
    for (auto& fc : classes_) {
      const bool internal = faces_[fc.face].internal;
      for (auto* M : {&fc.outflow, &fc.inflow}) {
        if (M == &fc.inflow && !internal)
          continue;
        auto grown = torch::zeros({fc.num_classes, ndim_, cap, cap}, F64);
        if (M->defined() && r0 > 0)
          grown.narrow(2, 0, r0).narrow(3, 0, r0).copy_(
            M->narrow(2, 0, r0).narrow(3, 0, r0));
        *M = grown;
      }
    }
    classes_capacity_ = cap;
  }
  // A class only selects points, so the products are summed over the
  // groups first: per point q, U_q = sum_g (T X)_{gq}^T (T P)_{gq} (r1, p),
  // shared by every class and dimension. The new columns of O_{c,i} are then
  // sum_q [q positive in c] n_{q,i} w_q U_q, one (C, Qf) x (Qf, r1 p)
  // product per dimension; O is symmetric (rows = transpose). For I, with
  // V_q = sum_g (T X)_{gq}^T (T^nb P)_{gq} and W_q = sum_g (T P)_{gq}^T
  // (T^nb [X P])_{gq}, the new columns and rows on the negative points.
  // The (C, Qf) x (Qf, r1 p) products on one thread: MKL's threaded kernels
  // for these shapes round differently per thread count, and a solve must
  // not depend on it (they are small; the samplers' threads cap their LA)
  linalg::lapack::SingleThreaded single;
  const auto J = Slice(r0, r1);
  for (auto& fc : classes_) {
    const auto& fp = faces_[fc.face];
    const int64_t Q = fc.moments.size(1), C = fc.num_classes;
    const auto PX = face_own(fp).reshape({basis_groups_, Q, r1});
    const auto PXJ = PX.narrow(2, r0, p);
    // Sums over the few groups as elementwise multiply-adds (no reduction
    // across threads: the same bits for any thread count)
    auto outer = [&](const torch::Tensor& L, const torch::Tensor& R) {
      auto out = L[0].unsqueeze(2) * R[0].unsqueeze(1);
      for (int64_t g = 1; g < basis_groups_; g++)
        out.addcmul_(L[g].unsqueeze(2), R[g].unsqueeze(1));
      return out.reshape({Q, -1});
    };
    const auto U = outer(PX, PXJ); // (Qf, r1 p)
    torch::Tensor V, W;
    if (fp.internal) {
      const auto NB = face_upwind(fp).reshape({basis_groups_, Q, r1});
      V = outer(PX, NB.narrow(2, r0, p)); // (Qf, r1 p)
      W = outer(PXJ, NB);                 // (Qf, p r1)
    }
    for (int64_t i = 0; i < ndim_; i++) {
      const auto m = fc.moments[i].unsqueeze(0); // (1, Qf)
      const auto col_O = (fc.positive * m).mm(U).view({C, r1, p});
      auto O = fc.outflow.select(1, i);
      O.narrow(1, 0, r1).narrow(2, r0, p).copy_(col_O);
      O.narrow(1, r0, p).narrow(2, 0, r1).copy_(col_O.transpose(1, 2));
      if (!fp.internal)
        continue;
      const auto neg = fc.negative * m;
      auto I = fc.inflow.select(1, i);
      I.narrow(1, 0, r1).narrow(2, r0, p).copy_(neg.mm(V).view({C, r1, p}));
      I.narrow(1, r0, p).narrow(2, 0, r1).copy_(neg.mm(W).view({C, p, r1}));
    }
  }
}

// =================================================================
// Basis lifecycle and bordering
// =================================================================

// Drop the basis and every projection (capacities are kept), and the
// gathered sources: the next extend() starts a new transport solve.
void ProjectedTransport::reset()
{
  rank_ = 0;
  basis_.resize(0, basis_groups_ * num_space_);
  for (auto& C : collision_)
    C.resize(0, 0);
  for (auto& s : streaming_)
    s.resize(0, 0);
  source_.resize(num_moments_ * static_cast<int64_t>(collision_.size()), 0);
  traces_own_.resize(num_trace_rows_, 0);
  if (any_internal_)
    traces_upwind_.resize(num_trace_rows_, 0);
  source_all_ = torch::Tensor();
  for (auto& F : outflow_)
    F.resize(0, 0);
  for (auto& set : inflow_)
    for (auto& K : set)
      K.resize(0, 0);
  for (auto& D : curved_cache_)
    D.resize(0, 0);
}

// The current basis, (r, G, Ns), a view of the stored rows.
torch::Tensor ProjectedTransport::basis() const
{
  return basis_.view().view({rank_, basis_groups_, num_space_});
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
  if (X.dim() != 3 || X.size(1) != basis_groups_ || X.size(2) != num_space_)
    throw utils::runtime_error(
      ctx("extend"), "X must be (p, basis_groups, fiber_length)");
  const int64_t p = X.size(0);
  if (p == 0)
    return;
  const int64_t r0 = rank_, r1 = rank_ + p;
  const auto J = Slice(r0, r1);

  // Append P to the basis rows; `all` is [X P], `added` is P
  basis_.resize(r1, basis_groups_ * num_space_);
  basis_.view().narrow(0, r0, p).copy_(X.reshape({p, -1}));
  const auto all =
    basis_.view().view({r1, basis_groups_, num_space_}); // (r1, G, Ns)
  const auto added = all.narrow(0, r0, p);               // (p, G, Ns)

  // The products below on one MKL thread: its threaded kernels round
  // differently per thread count, and a solve must not depend on it
  linalg::lapack::SingleThreaded single;
  // G the groups of a basis vector; Gc the collision blocks: one (summed
  // over the set's groups) for direction sampling, one per group of the set
  // with spatial_basis (sigma_{t,g} M, the only group-dependent volume term)
  const int64_t G = basis_groups_, Ns = num_space_, GN = G * Ns;
  const int64_t Gc = static_cast<int64_t>(collision_.size());
  const auto Xall = all.reshape({r1, GN});
  const auto Pvec = added.reshape({p, GN}).contiguous(); // the new vectors
  const double* pv = Pvec.data_ptr<double>();

  // Volume terms, every unit at once through the fiber's pair lists, in one
  // pass over the pairs: for each new vector P_j and group g the stack
  //   [ sigma M P | G_1 P .. G_ndim P | G_1^T P .. G_ndim^T P ]
  // (the sum over the set's groups is in the product below: each basis
  // vector spans all of them, sum_g P_g^T B_g X_g). Collision C =
  // blockdiag_u(sigma_{t,u,g} M_u) is symmetric: the new rows are the
  // transpose of the new columns. Streaming G_i(a, b) = int R_b d_i R_a (the
  // integrated-by-parts volume term) is not: the new columns X^T G_i P and
  // the new rows P^T G_i X = (G_i^T P)^T X both need an application. One
  // GEMM X [stack]^T then reads X once for all of them
  // stack[k][j] is a (G Ns) vector: k < Gc sigma M P_j (with spatial_basis,
  // block k for group k), k = Gc..Gc+ndim-1 G_i P_j, k = Gc+ndim.. G_i^T
  // P_j. Each pair (a, b) = (test, trial) adds to row a of the forward
  // products and to row b of the transposed ones
  const int64_t K = Gc + 2 * ndim_;
  auto stack = torch::zeros({K, p, GN}, F64);
  {
    double* st = stack.data_ptr<double>();
    const int64_t P = volume_row_.numel();
    const int64_t* prow = volume_row_.data_ptr<int64_t>();
    const int64_t* pcol = volume_col_.data_ptr<int64_t>();
    const double* mass = volume_mass_.data_ptr<double>();
    const double* grad = volume_grad_.data_ptr<double>();
    const double* sigma = sigma_.data_ptr<double>();
    for (int64_t j = 0; j < p; j++)
      for (int64_t g = 0; g < G; g++) {
        const double* y = pv + j * GN + g * Ns;
        for (int64_t p_ = 0; p_ < P; p_++) {
          const int64_t a = prow[p_], b = pcol[p_];
          const double my = mass[p_] * y[b];
          // Collision: block 0 at group g's rows with sigma_g, or one block
          // per group of the set on the single spatial row
          for (int64_t c = 0; c < Gc; c++) {
            const int64_t gs = Gc > 1 ? c : g;
            st[(c * p + j) * GN + g * Ns + a] += sigma[gs * Ns + a] * my;
          }
          for (int64_t i = 0; i < ndim_; i++) {
            const double gv = grad[i * P + p_];
            st[((Gc + i) * p + j) * GN + g * Ns + a] += gv * y[b];
            st[((Gc + ndim_ + i) * p + j) * GN + g * Ns + b] += gv * y[a];
          }
        }
      }
  }
  // cols[:, k p + j] = [X P]^T stack[k][j]: the new columns of C and S_i,
  // and (transposed) the new rows of S_i
  const auto cols = Xall.mm(stack.view({K * p, GN}).t()); // (r1, K p)
  // Sources: X^T s_m per angular moment m (a vector, no rows to border),
  // gathered at the set's groups once per transport solve (the first
  // extension after reset()), so the basis belongs to this solve
  // (rows (m, g) per group with spatial_basis: the source layout (M, G, Ns)
  // read as (M G, Ns))
  gather_sources();
  const auto col_P = source_all_.reshape({num_moments_ * Gc, GN}).mm(Pvec.t());
  // Write the borders: new columns [:, J], then new rows [J, :] (the
  // corner is written twice with the same value)
  for (int64_t c = 0; c < Gc; c++) {
    const auto col_C = cols.narrow(1, c * p, p);
    collision_[c].resize(r1, r1);
    collision_[c].view().index_put_({Slice(), J}, col_C);
    collision_[c].view().index_put_({J, Slice()}, col_C.t());
  }
  for (int64_t i = 0; i < ndim_; i++) {
    streaming_[i].resize(r1, r1);
    streaming_[i].view().index_put_(
      {Slice(), J}, cols.narrow(1, (Gc + i) * p, p));
    streaming_[i].view().index_put_(
      {J, Slice()}, cols.narrow(1, (Gc + ndim_ + i) * p, p).t());
  }
  source_.resize(num_moments_ * Gc, r1);
  source_.view().index_put_({Slice(), J}, col_P);

  // Faces, first the traces: the new vectors' values at every face's
  // quadrature points, (T_f P)_{g,q} = sum_a R_a(q) P_g[dof(a)], written as
  // new columns J of the global traces (face f's rows row0 + g * Qf + q);
  // INTERNAL faces also the upwind neighbor's trace at the same points
  // (T_f^nb P): the inflow a DEPENDENCY edge brings in from the neighbor.
  // Every face integral below is then a dense product over points
  // traces[row, r0 + j] = sum_a value[row, a] * P_j[index[row, a]], written
  // straight into the Growing storage (leading dimension = its capacity)
  auto write_traces = [&](Growing& traces, const torch::Tensor& index,
                        const torch::Tensor& value) {
    traces.resize(num_trace_rows_, r1);
    double* T = traces.storage.data_ptr<double>();
    const int64_t ld = traces.storage.size(1);
    const int64_t A = value.size(1);
    const int64_t* idx = index.data_ptr<int64_t>();
    const double* val = value.data_ptr<double>();
    for (int64_t q = 0; q < num_trace_rows_; q++)
      for (int64_t j = 0; j < p; j++) {
        double t = 0.0;
        for (int64_t a = 0; a < A; a++)
          t += val[q * A + a] * pv[j * GN + idx[q * A + a]];
        T[q * ld + r0 + j] = t;
      }
  };
  write_traces(traces_own_, own_index_, own_value_);
  if (any_internal_)
    write_traces(traces_upwind_, upwind_index_, upwind_value_);

  // Flat faces (constant unit normal n): (Omega . n_q)_+- w_q =
  // (Omega . n)_+- |n_q| w_q factors, so the direction only scales fixed
  // matrices. Summed over the faces sharing a normal (one block of rows):
  //   F_n     = sum_f (T_f X)^T W_f (T_f X)      outflow, symmetric
  //   K_{a,n} = sum_f (T_f X)^T W_f (T_f^nb X)   inflow over the faces
  //             angle set a sweeps upwind first (DEPENDENCY), test on the
  //             own side and trial on the neighbor's: not symmetric
  // with W_f = diag(|n_q| w_q). reduced_system() weights them by
  // (Omega . n)_+ and (Omega . n)_-. Per normal, F's and every set's K
  // columns are one product [T X]^T [W T P, W_a T^nb P, ...] (the own block
  // read once), and the K rows one product [W_a T P, ...]^T [T^nb X]
  for (size_t n = 0; n < normals_.size(); n++) {
    const auto [row0, count] = normal_rows_[n];
    outflow_[n].resize(r1, r1);
    for (int64_t a = 0; a < num_sets_; a++)
      inflow_[a][n].resize(r1, r1);
    if (count == 0)
      continue;
    // PX = T [X P] on this normal's rows, PXJ = T P; `sets` the angle sets
    // with at least one DEPENDENCY face of this normal
    const auto PX = traces_own_.view().narrow(0, row0, count);
    const auto PXJ = PX.narrow(1, r0, p);
    std::vector<int64_t> sets;
    for (int64_t a = 0; a < num_sets_; a++)
      if (dependency_measure_[a][n].defined())
        sets.push_back(a);
    std::vector<torch::Tensor> rhs {PXJ * normal_measure_[n].unsqueeze(1)};
    std::vector<torch::Tensor> lhs;
    torch::Tensor NB;
    if (!sets.empty()) {
      NB = traces_upwind_.view().narrow(0, row0, count);
      const auto NBJ = NB.narrow(1, r0, p);
      for (int64_t a : sets) {
        const auto wd = dependency_measure_[a][n].unsqueeze(1);
        rhs.push_back(NBJ * wd);
        lhs.push_back(PXJ * wd);
      }
    }
    // col = [T X P]^T [W T P | W_a T^nb P ...]: F's new columns first, then
    // one block of K columns per set
    const auto col = PX.t().mm(torch::cat(rhs, 1)); // (r1, p (1 + |sets|))
    const auto col_F = col.narrow(1, 0, p);
    outflow_[n].view().index_put_({Slice(), J}, col_F);
    outflow_[n].view().index_put_({J, Slice()}, col_F.t());
    if (sets.empty())
      continue;
    // Sets that sweep none of these faces' neighbors first keep zeros
    // (lagged sets take the inflow from the buffers, buffered_rhs())
    const auto row = torch::cat(lhs, 1).t().mm(NB); // (p |sets|, r1)
    for (size_t k = 0; k < sets.size(); k++) {
      const int64_t a = sets[k];
      inflow_[a][n].view().index_put_(
        {Slice(), J}, col.narrow(1, (1 + static_cast<int64_t>(k)) * p, p));
      inflow_[a][n].view().index_put_(
        {J, Slice()}, row.narrow(0, static_cast<int64_t>(k) * p, p));
    }
  }
  // Curved faces by class (per-direction ones border lazily, update_curved())
  extend_classes(r0, r1);
  rank_ = r1;
}

// =================================================================
// Per-direction assembly and solves
// =================================================================

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
  for (size_t k : per_direction_) {
    const auto& fp = faces_[k];
    const auto& geom = S.patches_[fp.unit].disc->faces()[fp.face];
    // c_q = (Omega . n_q)_+- w_q, as the kernel's upwind weights, repeated
    // per group to match the trace rows g * Qf + q
    const auto dot = geom.normal.mv(omega);
    const auto out =
      (dot.clamp_min(0.0) * geom.weights).repeat({basis_groups_});
    // Outflow (symmetric): new columns [X P]^T W+ (T P); new rows their
    // transpose, accumulated as rows since `row` also carries the inflow
    const auto PX = face_own(fp);
    const auto wPXn = PX.index({Slice(), J}) * out.unsqueeze(1);
    col += PX.t().mm(wPXn);
    row += wPXn.t().mm(PX);
    if (!fp.internal || !fp.dependency[a])
      continue;
    // Dependency inflow (not symmetric, subtracted): new columns
    // [X P]^T W- (T^nb P) and new rows (T P)^T W- [T^nb X, T^nb P]
    const auto in =
      ((-dot).clamp_min(0.0) * geom.weights).repeat({basis_groups_});
    const auto NB = face_upwind(fp);
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
torch::Tensor ProjectedTransport::buffered_rhs(int64_t u_index) const
{
  if (u_index < 0 || u_index >= num_units_)
    throw utils::runtime_error(
      ctx("buffered_rhs"), unit_name(true) + " out of range");
  const auto& S = solver_;
  const int64_t d = unit_direction(u_index), g_unit = unit_group(u_index);
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
    // The basis on the face's dofs, (r, basis_groups, Nf); the unit's
    // groups: every group of the set (basis group bg), or its own (basis
    // group 0, spatial_basis)
    const auto X_face =
      X.narrow(2, S.fiber_offsets_[u], N).index_select(2, geom.dofs);
    const int64_t g0 = g_unit < 0 ? 0 : g_unit;
    const int64_t g1 = g_unit < 0 ? num_groups_ : g_unit + 1;
    for (int64_t bg = g0; bg < g1; bg++) {
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
      rhs += X_face.select(1, g_unit < 0 ? bg : 0).mv(load);
    }
  }
  return rhs;
}

// The reduced system of unit u (direction d, group g with spatial_basis),
//   A_u = C(_g) - sum_i Omega_i S_i
//       + sum_n [(Omega . n)_+ F_n - (Omega . n)_- K_{a,n}]     flat faces
//       + sum_{curved} (class terms or the cached D_d)          curved faces
//   b_u = sum_m Y_m(d) X^T s_m(,g) + X^T (buffered inflow of u),
// a = d's angle set. Only d's own curved cache is written, so units of
// different directions can be assembled concurrently.
std::pair<torch::Tensor, torch::Tensor> ProjectedTransport::reduced_system(
  int64_t u)
{
  if (u < 0 || u >= num_units_)
    throw utils::runtime_error(
      ctx("reduced_system"), unit_name(true) + " out of range");
  const auto& S = solver_;
  const int64_t d = unit_direction(u), g = unit_group(u);
  const int64_t a = S.graph_->direction_set().template data_ptr<int64_t>()[d];
  const auto omega = S.omega_[d];
  const double* w = omega.template data_ptr<double>();

  auto A = collision_[g < 0 ? 0 : g].view().clone();
  for (int64_t i = 0; i < ndim_; i++)
    A -= w[i] * streaming_[i].view();
  for (size_t n = 0; n < normals_.size(); n++) {
    const double dot = normals_[n].dot(omega).template item<double>();
    if (dot > 0.0)
      A += dot * outflow_[n].view();
    else if (dot < 0.0)
      A += dot * inflow_[a][n].view(); // -(Omega . n)_- K
  }
  // Curved faces per sign class: sum_i Omega_i (O_{c,i} + I_{c,i}), I
  // for the angle sets that sweep the face's neighbor first
  const int64_t r = rank_;
  for (const auto& fc : (r > 0 ? classes_ : std::vector<FaceClasses> {})) {
    const int64_t c = fc.class_of[d];
    const auto& fp = faces_[fc.face];
    const bool inflow = fp.internal && fp.dependency[a];
    for (int64_t i = 0; i < ndim_; i++) {
      A.add_(fc.outflow[c][i].narrow(0, 0, r).narrow(1, 0, r), w[i]);
      if (inflow)
        A.add_(fc.inflow[c][i].narrow(0, 0, r).narrow(1, 0, r), w[i]);
    }
  }
  if (!per_direction_.empty()) {
    update_curved(d);
    A += curved_cache_[d].view();
  }
  // X^T s_m of the unit's group: rows (m, g) of source_ with spatial_basis
  const auto src =
    g < 0 ? source_.view()
          : source_.view().view({num_moments_, num_groups_, r}).select(1, g);
  auto b = S.harmonics_[d].matmul(src) + buffered_rhs(u);
  return {A, b};
}

// The reduced solutions c_d = A_d^{-1} b_d, (D, r): assembly, LU and solve
// per direction, in parallel over directions (an LU per direction: A_d is
// not symmetric and differs per direction).
torch::Tensor ProjectedTransport::solve(const std::vector<int64_t>& units)
{
  const auto& directions = units;
  const int64_t D = static_cast<int64_t>(units.size());
  const int64_t r = rank_;
  std::vector<uint8_t> seen(static_cast<size_t>(num_units_), 0);
  for (int64_t u : units) {
    if (u < 0 || u >= num_units_)
      throw utils::runtime_error(
        ctx("solve"), unit_name(true) + " out of range");
    if (seen[u]++)
      throw utils::runtime_error(
        ctx("solve"), unit_name(true) + "s must be distinct");
  }
  auto C = torch::zeros({D, r}, F64);
  if (D == 0 || r == 0)
    return C;
  // The per-direction curved caches first, each by one task (several units
  // can share a direction with spatial_basis): the assembly below then only
  // reads them, so units are assembled concurrently
  if (!per_direction_.empty()) {
    std::vector<int64_t> dirs;
    std::vector<uint8_t> has(static_cast<size_t>(num_directions_), 0);
    for (int64_t u : units)
      if (!has[unit_direction(u)]++)
        dirs.push_back(unit_direction(u));
    at::parallel_for(
      0, static_cast<int64_t>(dirs.size()), 1, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; i++)
          update_curved(dirs[i]);
      });
  }
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
      throw utils::runtime_error(
        ctx("solve"), "The reduced matrix of " + unit_name(false) + " " +
                        std::to_string(directions[i]) +
                        " is singular (rank-deficient basis?)");
  return C;
}

// =================================================================
// Full-space residuals
// =================================================================

// The residuals r_d = b_d - A_d X c_d of a batch of directions, matrix-free
// on the whole fiber with the operators extend() projects (no per-unit
// kernel calls): with Y = C X (one row per direction),
//   A_d y = sigma M y - sum_i Omega_i G_i y
//         + T^T diag((Omega_d . n_q)_+ w_q) T y
//         - T^T diag((-Omega_d . n_q)_+ w_q [DEPENDENCY for set(d)]) T^nb y
// over every face point (flat and curved alike; n_q unnormalized), and b_d
// = sum_m Y_m(d) s_m + the buffered inflow (lagged edges, reflective faces)
// as the kernel assembles it. This is SweepSolver::transport_residual() of
// X c_d (the sweep's L, source and inflow), unless Options::implicit_lagged
// moves the lagged inflow into A_d.
torch::Tensor ProjectedTransport::residuals(
  const std::vector<int64_t>& units, const torch::Tensor& C) const
{
  const int64_t D = static_cast<int64_t>(units.size());
  const int64_t r = rank_;
  if (C.dim() != 2 || C.size(0) != D || C.size(1) != r)
    throw utils::runtime_error(
      ctx("residuals"), "C must be (len(units), rank)");
  for (int64_t u : units)
    if (u < 0 || u >= num_units_)
      throw utils::runtime_error(
        ctx("residuals"), unit_name(true) + " out of range");
  // G the groups of a residual row: the set's (directions) or 1 (pairs)
  const int64_t G = basis_groups_, Ns = num_space_;
  if (D == 0)
    return torch::empty({0, G, Ns}, F64);
  const auto& S = solver_;
  const bool pairs = options_.spatial_basis;
  // One MKL thread: a residual must not depend on the thread count
  linalg::lapack::SingleThreaded single;
  std::vector<int64_t> dirs(units.size()), unit_groups(units.size(), 0);
  for (size_t i = 0; i < units.size(); i++) {
    dirs[i] = unit_direction(units[i]);
    unit_groups[i] = pairs ? unit_group(units[i]) : 0;
  }
  const auto& directions = dirs;
  const auto dir_index = torch::tensor(dirs, torch::kInt64);
  const auto omega = S.omega_.to(F64).index_select(0, dir_index); // (D, ndim)

  // b_u: the sources (of the unit's group with spatial_basis), then the
  // buffered inflow
  gather_sources();
  const auto Yd = S.harmonics_.to(F64).index_select(0, dir_index); // (D, M)
  auto R = pairs ? torch::einsum("dm,mdn->dn",
                     {Yd, source_all_.index_select(
                            1, torch::tensor(unit_groups, torch::kInt64))})
                     .contiguous() // (D, Ns)
                 : Yd.mm(source_all_.reshape({num_moments_, G * Ns}))
                     .contiguous(); // (D, G Ns)
  {
    double* out = R.data_ptr<double>();
    for (int64_t i = 0; i < D; i++)
      add_buffered_inflow(units[i], out + i * G * Ns);
  }
  if (r == 0)
    return R.view({D, G, Ns});

  // y = X c_d
  const auto Y = C.to(F64).contiguous().mm(basis_.view()).contiguous();

  // r -= A_d y, one direction per task (each writes only its own row, so
  // any thread count gives the same bits), in plain loops over the fiber's
  // pairs and trace rows: one pass each, no temporaries (torch index ops
  // over these arrays cost several passes; per-unit kernel calls cost a
  // setup per unit, which dominates on many small units)
  // Raw views of everything the loop reads: y and r rows per direction, the
  // pair lists (volume), and per trace row its gather indices and values
  // (own side and upwind), point normal n_q, weight w_q and per-set
  // DEPENDENCY flag
  const double* y_all = Y.data_ptr<double>();
  double* r_all = R.data_ptr<double>();
  const double* om_all = omega.contiguous().data_ptr<double>();
  const int64_t P = volume_row_.numel();
  const int64_t* prow = volume_row_.data_ptr<int64_t>();
  const int64_t* pcol = volume_col_.data_ptr<int64_t>();
  const double* mass = volume_mass_.data_ptr<double>();
  const double* grad = volume_grad_.data_ptr<double>(); // (ndim, P)
  const double* sigma = sigma_.data_ptr<double>();      // (G, Ns)
  const int64_t rows = num_trace_rows_;
  const int64_t A = own_value_.size(1);
  const int64_t* oidx = own_index_.data_ptr<int64_t>();
  const double* oval = own_value_.data_ptr<double>();
  const int64_t* uidx =
    any_internal_ ? upwind_index_.data_ptr<int64_t>() : nullptr;
  const double* uval =
    any_internal_ ? upwind_value_.data_ptr<double>() : nullptr;
  const double* nrm = row_normal_.data_ptr<double>();
  const double* wq = row_weight_.data_ptr<double>();
  const double* dep_all = row_dependency_.data_ptr<double>();
  const int64_t* set_of =
    S.graph_->direction_set().template data_ptr<int64_t>();
  const int64_t nd = ndim_;
  at::parallel_for(0, D, 1, [&](int64_t j0, int64_t j1) {
    for (int64_t j = j0; j < j1; j++) {
      const double* y = y_all + j * G * Ns;
      double* r = r_all + j * G * Ns;
      const double* om = om_all + j * nd;
      // Volume: (sigma M - sum_i Omega_i G_i) y, group by group (with
      // spatial_basis one row, sigma of the unit's group)
      for (int64_t g = 0; g < G; g++) {
        const double* yg = y + g * Ns;
        double* rg = r + g * Ns;
        const double* sg = sigma + (pairs ? unit_groups[j] : g) * Ns;
        for (int64_t p = 0; p < P; p++) {
          double v = sg[prow[p]] * mass[p];
          for (int64_t i = 0; i < nd; i++)
            v -= om[i] * grad[i * P + p];
          rg[prow[p]] -= v * yg[pcol[p]];
        }
      }
      // Faces: per trace row the upwind weights at its point, the own trace
      // (outflow) and the neighbor's (DEPENDENCY inflow), scattered back
      // onto the own side's test functions
      const double* dep = dep_all + set_of[directions[j]] * rows;
      for (int64_t q = 0; q < rows; q++) {
        double dot = 0.0;
        for (int64_t i = 0; i < nd; i++)
          dot += om[i] * nrm[q * nd + i];
        // Outflow at this point if Omega . n_q > 0; inflow from the upwind
        // neighbor if < 0 and the face is a DEPENDENCY edge of d's set
        // (other inflow is buffered: already in b_d)
        const double w_out = dot > 0.0 ? dot * wq[q] : 0.0;
        const double w_in =
          uidx && dot < 0.0 && dep[q] != 0.0 ? -dot * wq[q] : 0.0;
        if (w_out == 0.0 && w_in == 0.0)
          continue;
        const int64_t* oi = oidx + q * A;
        const double* ov = oval + q * A;
        double f = 0.0;
        if (w_out != 0.0) {
          double t = 0.0;
          for (int64_t a = 0; a < A; a++)
            t += ov[a] * y[oi[a]];
          f += w_out * t;
        }
        if (w_in != 0.0) {
          const int64_t* ui = uidx + q * A;
          const double* uv = uval + q * A;
          double t = 0.0;
          for (int64_t a = 0; a < A; a++)
            t += uv[a] * y[ui[a]];
          f -= w_in * t;
        }
        // Test against the own side's basis functions at this point (T^T)
        for (int64_t a = 0; a < A; a++)
          r[oi[a]] -= ov[a] * f;
      }
    }
  });
  return R.view({D, G, Ns});
}

} // namespace ttnte::sweep
