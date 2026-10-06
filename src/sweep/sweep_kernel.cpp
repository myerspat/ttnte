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
  const bool banded = block.solver == LocalSolverType::BANDED_LU;
  const int64_t matrix_size = banded ? N * v.ldab : N * N;
  const int64_t* index = banded ? v.band_index : v.dense_index;

  ws.values.resize(P);
  ws.matrix.resize(matrix_size);
  ws.rhs.resize(N);
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

    // Streaming + outflow pair values (group independent)
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

    // Inflow weights per face (group independent)
    std::array<bool, MAX_FACES> active {};
    for (int64_t f = 0; f < num_faces; f++) {
      const auto& in = unit.inflow[f];
      double* c = c_in + f * Qmax;
      if (in.kind == FaceInflow::Kind::NEIGHBOR) {
        active[f] = upwind(v.faces[f], omega, ndim, -1.0, 1.0, c);
      } else if (in.kind == FaceInflow::Kind::REFLECTIVE) {
        // The TT operator's reflective inflow uses the reflected
        // direction's outflow mask
        active[f] = upwind(v.faces[f], block.omega + in.reflected[d] * ndim,
          ndim, 1.0, in.albedo, c);
      }
    }

    const double* Y = block.harmonics + d * M;
    const double* Yw = block.weighted_harmonics + d * M;
    for (int64_t bg = 0; bg < block.num_block_groups; bg++) {
      const int64_t g = block.groups[bg];

      // Operator
      std::fill(A, A + matrix_size, 0.0);
      const double sigma = unit.sigma_t[g];
      for (int64_t p = 0; p < P; p++)
        A[index[p]] = values[p] + sigma * v.mass[p];

      // Source: sum_m Y_m(d) s_{m,g}
      std::fill(b, b + N, 0.0);
      for (int64_t m = 0; m < M; m++) {
        const double y = Y[m];
        if (y == 0.0)
          continue;
        const double* s = unit.source_moments + (m * G + g) * N;
        for (int64_t i = 0; i < N; i++)
          b[i] += y * s[i];
      }

      // Upwind inflow: b_v += sum_q c_q R_v(q) R_u(q) trace_u
      for (int64_t f = 0; f < num_faces; f++) {
        if (!active[f])
          continue;
        const auto& in = unit.inflow[f];
        const auto& face = v.faces[f];
        const double* c = c_in + f * Qmax;
        const int64_t src_dir =
          in.kind == FaceInflow::Kind::REFLECTIVE ? in.reflected[d] : d;
        const double* trace =
          in.trace + (src_dir * G + g) * in.source_face_dofs;
        for (int64_t k = 0; k < face.num_pairs; k++) {
          const int64_t u =
            in.trace_map ? in.trace_map[face.pair_b[k]] : face.pair_b[k];
          b[face.dofs[face.pair_a[k]]] +=
            c[face.pair_q[k]] * face.pair_val[k] * trace[u];
        }
      }

      // Solve
      int64_t info;
      if (banded) {
        info = linalg::banded::lu_factor(A, v.ldab, N, v.kl, v.ku, ipiv);
        if (info == 0)
          linalg::banded::lu_solve(A, v.ldab, N, v.kl, v.ku, ipiv, b);
      } else {
        info = linalg::dense::lu_factor(A, N, ipiv);
        if (info == 0)
          linalg::dense::lu_solve(A, N, ipiv, b);
      }
      if (info != 0) {
        throw utils::runtime_error("ttnte::sweep::sweep_unit",
          "Singular local system (direction " + std::to_string(d) + ", group " +
            std::to_string(g) + ")");
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
      for (int64_t m = 0; m < M; m++) {
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

void scattering_source(const KernelView& view, const double* scatter,
  int64_t num_orders, const int64_t* moment_order, int64_t num_moments,
  int64_t num_groups, const double* moments, const double* external,
  double* out, std::vector<double>& scratch)
{
  const int64_t N = view.num_dofs;
  const int64_t G = num_groups;
  scratch.resize(G * N);
  double* t = scratch.data();
  std::fill(out, out + num_moments * G * N, 0.0);

  for (int64_t m = 0; m < num_moments; m++) {
    const int64_t l = moment_order[m];
    double* dst = out + m * G * N;
    if (scatter && l < num_orders) {
      // t = sigma_l phi_m  (G x G times G x N)
      std::fill(t, t + G * N, 0.0);
      const double* sig = scatter + l * G * G;
      const double* phi = moments + m * G * N;
      for (int64_t g = 0; g < G; g++) {
        for (int64_t gp = 0; gp < G; gp++) {
          const double s = sig[g * G + gp];
          if (s == 0.0)
            continue;
          for (int64_t i = 0; i < N; i++)
            t[g * N + i] += s * phi[gp * N + i];
        }
      }
      // dst = M t
      for (int64_t g = 0; g < G; g++) {
        const double* tg = t + g * N;
        double* dg = dst + g * N;
        for (int64_t p = 0; p < view.num_pairs; p++)
          dg[view.pair_row[p]] += view.mass[p] * tg[view.pair_col[p]];
      }
    }
    if (m == 0 && external) {
      for (int64_t i = 0; i < G * N; i++)
        dst[i] += external[i];
    }
  }
}

} // namespace ttnte::sweep
