#include "ttnte/sweep/sweep_kernel.hpp"
#include "ttnte/linalg/banded_lu.hpp"
#include "ttnte/linalg/dense_lu.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>

namespace ttnte::sweep {

namespace {

/// @brief Clipped, weighted normal components of one face for one
/// direction: c_q = scale * max(sign * Omega . n_q, 0) w_q.
/// @return Whether any c_q is nonzero.
bool upwind(const FaceView& face, const double* omega, int64_t ndim,
  double sign, double scale, double* c)
{
  bool any = false;
  for (int64_t q = 0; q < face.num_points; q++) {
    double dot = 0.0;
    for (int64_t i = 0; i < ndim; i++)
      dot += omega[i] * face.normal[q * ndim + i];
    dot *= sign;
    c[q] = dot > 0.0 ? scale * dot * face.weights[q] : 0.0;
    any |= c[q] != 0.0;
  }
  return any;
}

/// @brief Group-independent pair values of L_d without the collision term:
/// values_p = -Omega_d . G_p plus, on every outflow face,
/// sum_q c_q R_a(q) R_b(q) at the face pairs' volume positions.
/// @param c_out (max face points) scratch.
void streaming_values(
  const KernelView& v, const double* omega, double* values, double* c_out)
{
  const int64_t ndim = v.ndim;
  const int64_t P = v.num_pairs;
  for (int64_t p = 0; p < P; p++) {
    double s = 0.0;
    for (int64_t i = 0; i < ndim; i++)
      s -= omega[i] * v.grad[i * P + p];
    values[p] = s;
  }
  for (const auto& face : v.faces) {
    if (!upwind(face, omega, ndim, 1.0, 1.0, c_out))
      continue;
    for (int64_t k = 0; k < face.num_pairs; k++)
      values[face.pair_volume_index[k]] +=
        c_out[face.pair_q[k]] * face.pair_val[k];
  }
}

/// @brief Inflow weights of face f for direction d (group independent):
/// c = (Omega_d . n)_- w on a NEIGHBOR face, and the reflected direction's
/// outflow mask times the albedo on a REFLECTIVE one.
/// @param c (max face points) output.
/// @return Whether the face has any inflow.
bool face_inflow_weights(const UnitSweep& unit, const SweepBlock& block,
  int64_t d, size_t f, double* c)
{
  const auto& v = unit.disc->view();
  const int64_t ndim = v.ndim;
  const auto& in = unit.inflow[f];
  if (in.kind == FaceInflow::Kind::NEIGHBOR)
    return upwind(v.faces[f], block.omega + d * ndim, ndim, -1.0, 1.0, c);
  if (in.kind == FaceInflow::Kind::REFLECTIVE) {
    // The TT operator's reflective inflow uses the reflected direction's
    // outflow mask
    return upwind(v.faces[f], block.omega + in.reflected[d] * ndim, ndim, 1.0,
      in.albedo, c);
  }
  return false;
}

/// @brief face_inflow_weights() of every face.
/// @param c_in (faces, max face points) output.
/// @param active Whether each face has any inflow.
void inflow_weights(const UnitSweep& unit, const SweepBlock& block, int64_t d,
  double* c_in, std::array<bool, MAX_FACES>& active)
{
  const auto& v = unit.disc->view();
  active.fill(false);
  for (size_t f = 0; f < v.faces.size(); f++)
    active[f] =
      face_inflow_weights(unit, block, d, f, c_in + f * v.max_face_points);
}

/// @brief b = sum_m Y_m s_{m,g}: the scattering + external source of
/// (direction, group g) (b overwritten).
void add_source(const UnitSweep& unit, const double* Y, int64_t M, int64_t G,
  int64_t g, int64_t N, double* b)
{
  std::fill(b, b + N, 0.0);
  for (int64_t m = 0; m < M; m++) {
    const double y = Y[m];
    if (y == 0.0)
      continue;
    const double* s = unit.source_moments + (m * G + g) * N;
    for (int64_t i = 0; i < N; i++)
      b[i] += y * s[i];
  }
}

/// @brief Upwind inflow of one face: b_v += sum_q c_q R_v(q) R_u(q)
/// trace_u, trace the source face's values (num source-face dofs) for the
/// (direction, group).
void add_inflow(const FaceView& face, const FaceInflow& in, const double* c,
  const double* trace, double* b)
{
  if (in.coupling_q) {
    for (int64_t k = 0; k < in.num_coupling_pairs; k++)
      b[face.dofs[in.coupling_a[k]]] +=
        c[in.coupling_q[k]] * in.coupling_val[k] * trace[in.coupling_b[k]];
    return;
  }
  for (int64_t k = 0; k < face.num_pairs; k++) {
    const int64_t u =
      in.trace_map ? in.trace_map[face.pair_b[k]] : face.pair_b[k];
    b[face.dofs[face.pair_a[k]]] +=
      c[face.pair_q[k]] * face.pair_val[k] * trace[u];
  }
}

/// @return Group bg's values of a (num_groups, stride) array.
template<typename T>
T* group_at(T* values, int64_t bg, int64_t stride) noexcept
{
  return values + bg * stride;
}

/// @return The lagged or reflected trace a face reads for (direction d,
/// group g): the source face's values in the FaceInflow's buffer.
const double* inflow_trace(
  const FaceInflow& in, int64_t d, int64_t g, int64_t G)
{
  const int64_t src_dir =
    in.kind == FaceInflow::Kind::REFLECTIVE ? in.reflected[d] : d;
  return in.trace + (src_dir * G + g) * in.source_face_dofs;
}

} // namespace

void sweep_unit(
  const UnitSweep& unit, const SweepBlock& block, SweepWorkspace& ws)
{
  const auto& v = unit.disc->view();
  const int64_t ndim = v.ndim;
  const int64_t N = v.num_dofs;
  const int64_t P = v.num_pairs;
  const int64_t M = block.num_moments;
  const int64_t G = block.num_groups;
  const int64_t num_faces = static_cast<int64_t>(v.faces.size());
  const int64_t Qmax = v.max_face_points;
  const bool banded = block.solver != LocalSolverType::DENSE_LU;
  // Direct structured solvers assemble their own (smaller) systems: no
  // patch matrix, no factor cache
  const bool structured = unit.kronecker != nullptr || unit.extruded != nullptr;
  const int64_t matrix_size = banded ? N * v.ldab : N * N;
  const int64_t* index = banded ? v.band_index : v.dense_index;

  ws.values.resize(P);
  ws.matrix.resize(matrix_size);
  ws.rhs.resize(N);
  ws.permuted.resize(N);
  ws.inflow.resize((num_faces + 1) * Qmax);
  ws.pivots.resize(N);
  double* values = ws.values.data();
  double* A = ws.matrix.data();
  double* b = ws.rhs.data();
  double* c_in = ws.inflow.data();         // (faces, Qmax)
  double* c_out = c_in + num_faces * Qmax; // (Qmax) scratch
  int64_t* ipiv = ws.pivots.data();

  for (int64_t bd = 0; bd < block.num_block_directions; bd++) {
    const int64_t d = block.directions[bd];
    const double* omega = block.omega + d * ndim;
    // Every group of this direction cached: no operator values needed
    auto ready = [&](int64_t g) {
      return unit.factors && unit.factor_ready[d * G + g];
    };
    bool cached = structured;
    if (!cached && unit.factors) {
      cached = true;
      for (int64_t bg = 0; bg < block.num_block_groups && cached; bg++)
        cached = ready(block.groups[bg]);
    }

    // Streaming + outflow pair values (group independent), not needed when
    // the factors are cached
    if (!cached)
      streaming_values(v, omega, values, c_out);

    // Inflow weights per face (group independent)
    std::array<bool, MAX_FACES> active {};
    inflow_weights(unit, block, d, c_in, active);

    const double* Y = block.harmonics + d * M;
    const double* Yw = block.weighted_harmonics + d * M;
    for (int64_t bg = 0; bg < block.num_block_groups; bg++) {
      const int64_t g = block.groups[bg];

      // Operator: assembled and factored here, or read from the cache
      double* L = A;
      int64_t* piv = ipiv;
      if (unit.factors && !structured && !unit.iterative) {
        L = unit.factors + (d * G + g) * matrix_size;
        piv = unit.factor_pivots + (d * G + g) * N;
      }
      auto assemble_and_factor = [&]() {
        std::fill(L, L + matrix_size, 0.0);
        const double sigma = unit.sigma_t[g];
        for (int64_t p = 0; p < P; p++)
          L[index[p]] = values[p] + sigma * v.mass[p];
        const int64_t info =
          banded ? linalg::banded::lu_factor(L, v.ldab, N, v.kl, v.ku, piv)
                 : linalg::dense::lu_factor(L, N, piv);
        if (info != 0) {
          throw utils::runtime_error("ttnte::sweep::sweep_unit",
            "Singular local system (direction " + std::to_string(d) +
              ", group " + std::to_string(g) + ")");
        }
      };
      if (!structured && !unit.iterative && !ready(g)) {
        assemble_and_factor();
        if (unit.factors)
          unit.factor_ready[d * G + g] = 1;
      }

      // Source: sum_m Y_m(d) s_{m,g}, and the upwind inflow
      add_source(unit, Y, M, G, g, N, b);
      for (int64_t f = 0; f < num_faces; f++)
        if (active[f])
          add_inflow(v.faces[f], unit.inflow[f], c_in + f * Qmax,
            inflow_trace(unit.inflow[f], d, g, G), b);

      // Solve: the structured solver of the unit if it has one, LU when it
      // has none or reports failure (an unconverged iterative solve)
      bool solved = false;
      if (unit.iterative) {
        ws.saved.assign(b, b + N);
        solved = unit.iterative->solve(omega, unit.sigma_t[g], b, ws.iterative);
        if (!solved) {
          std::copy(ws.saved.begin(), ws.saved.end(), b);
          assemble_and_factor();
        }
      } else if (unit.kronecker) {
        solved = unit.kronecker->solve(omega, unit.sigma_t[g], b, ws.kronecker);
      } else if (unit.extruded) {
        solved = unit.extruded->solve(omega, unit.sigma_t[g], b, ws.extruded);
      }
      if (!solved) {
        if (banded && v.band_order) {
          double* x = ws.permuted.data();
          for (int64_t j = 0; j < N; j++)
            x[j] = b[v.band_order[j]];
          linalg::banded::lu_solve(L, v.ldab, N, v.kl, v.ku, piv, x);
          for (int64_t j = 0; j < N; j++)
            b[v.band_order[j]] = x[j];
        } else if (banded)
          linalg::banded::lu_solve(L, v.ldab, N, v.kl, v.ku, piv, b);
        else
          linalg::dense::lu_solve(L, N, piv, b);
      }

      // Outgoing traces, angular flux and moments
      for (int64_t f = 0; f < num_faces; f++) {
        double* out = unit.trace_out[f];
        if (!out)
          continue;
        const auto& face = v.faces[f];
        double* dst = out + (d * G + g) * face.num_dofs;
        for (int64_t k = 0; k < face.num_dofs; k++)
          dst[k] = b[face.dofs[k]];
      }
      if (unit.psi) {
        std::copy(b, b + N, unit.psi + (d * G + g) * N);
      }
      if (unit.fibers) {
        std::copy(b, b + N,
          unit.fibers + (unit.fiber_slot[bd] + bg) * unit.fiber_stride);
      }
      for (int64_t m = 0; m < M && unit.moments; m++) {
        const double y = Yw[m];
        if (y == 0.0)
          continue;
        double* dst = unit.moments + (m * block.num_block_groups + bg) * N;
        for (int64_t i = 0; i < N; i++)
          dst[i] += y * b[i];
      }
    }
  }
}

bool buffered_inflow(const UnitSweep& unit, const SweepBlock& block, int64_t d,
  int64_t g, size_t f, double* c, double* b)
{
  if (!face_inflow_weights(unit, block, d, f, c))
    return false;
  const auto& in = unit.inflow[f];
  add_inflow(unit.disc->view().faces[f], in, c,
    inflow_trace(in, d, g, block.num_groups), b);
  return true;
}

void unit_residual(const UnitSweep& unit, const SweepBlock& block,
  const double* x, int64_t stride,
  const std::array<const double*, MAX_FACES>& upstream, double* r,
  SweepWorkspace& ws)
{
  if (block.num_block_directions != 1)
    throw utils::runtime_error(
      "ttnte::sweep::unit_residual", "One direction per call");
  const auto& v = unit.disc->view();
  const int64_t N = v.num_dofs;
  const int64_t P = v.num_pairs;
  const int64_t M = block.num_moments;
  const int64_t G = block.num_groups;
  const int64_t num_faces = static_cast<int64_t>(v.faces.size());
  const int64_t Qmax = v.max_face_points;
  const int64_t d = block.directions[0];
  const double* omega = block.omega + d * v.ndim;

  ws.values.resize(P);
  ws.inflow.resize((num_faces + 1) * Qmax);
  double* values = ws.values.data();
  double* c_in = ws.inflow.data();
  double* c_out = c_in + num_faces * Qmax;

  // The operator and inflow weights of the sweep (sweep_unit)
  streaming_values(v, omega, values, c_out);
  std::array<bool, MAX_FACES> active {};
  inflow_weights(unit, block, d, c_in, active);

  const double* Y = block.harmonics + d * M;
  for (int64_t bg = 0; bg < block.num_block_groups; bg++) {
    const int64_t g = block.groups[bg];
    const double* xg = group_at(x, bg, stride);
    double* rg = group_at(r, bg, stride);
    // r = source + inflow (from x upstream where given, else the buffers)
    add_source(unit, Y, M, G, g, N, rg);
    for (int64_t f = 0; f < num_faces; f++) {
      if (!active[f])
        continue;
      const auto& in = unit.inflow[f];
      const double* trace = upstream[f]
                              ? group_at(upstream[f], bg, in.source_face_dofs)
                              : inflow_trace(in, d, g, G);
      add_inflow(v.faces[f], in, c_in + f * Qmax, trace, rg);
    }
    // r -= L x, L the pair values plus sigma_t M
    const double sigma = unit.sigma_t[g];
    for (int64_t p = 0; p < P; p++)
      rg[v.pair_row[p]] -= (values[p] + sigma * v.mass[p]) * xg[v.pair_col[p]];
  }
}

void transfer_source(const KernelView& view, const double* transfer,
  int64_t num_orders, const int64_t* moment_order, int64_t num_moments,
  int64_t num_groups, const double* moments, double scale,
  const int64_t* dest_groups, int64_t num_dest, const uint8_t* source_mask,
  double* out, std::vector<double>& scratch)
{
  const int64_t N = view.num_dofs;
  const int64_t G = num_groups;
  scratch.resize(N);
  double* t = scratch.data();

  for (int64_t m = 0; m < num_moments; m++) {
    const int64_t l = moment_order[m];
    if (l >= num_orders)
      continue;
    const double* T = transfer + l * G * G;
    const double* phi = moments + m * G * N;
    for (int64_t k = 0; k < num_dest; k++) {
      const int64_t g = dest_groups[k];
      // t = scale * sum_g' T_{g,g'} phi_g'
      std::fill(t, t + N, 0.0);
      bool any = false;
      for (int64_t gp = 0; gp < G; gp++) {
        const double c = T[g * G + gp];
        if (c == 0.0 || (source_mask && !source_mask[gp]))
          continue;
        any = true;
        const double sc = scale * c;
        const double* pg = phi + gp * N;
        for (int64_t i = 0; i < N; i++)
          t[i] += sc * pg[i];
      }
      if (!any)
        continue;
      // out += M t
      double* dst = out + (m * G + g) * N;
      for (int64_t p = 0; p < view.num_pairs; p++)
        dst[view.pair_row[p]] += view.mass[p] * t[view.pair_col[p]];
    }
  }
}

double transfer_total(const KernelView& view, const double* transfer,
  int64_t num_groups, const double* scalar_flux, const double* basis_integral,
  const uint8_t* source_mask)
{
  // sum_v (M phi)_v = sum_u (int R_u) phi_u (partition of unity)
  const int64_t N = view.num_dofs;
  const int64_t G = num_groups;
  double total = 0.0;
  for (int64_t gp = 0; gp < G; gp++) {
    if (source_mask && !source_mask[gp])
      continue;
    double column = 0.0;
    for (int64_t g = 0; g < G; g++)
      column += transfer[g * G + gp];
    if (column == 0.0)
      continue;
    double integral = 0.0;
    for (int64_t u = 0; u < N; u++)
      integral += basis_integral[u] * scalar_flux[gp * N + u];
    total += column * integral;
  }
  return total;
}

} // namespace ttnte::sweep
