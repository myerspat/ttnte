#include "ttnte/sweep/extruded_solver.hpp"
#include "ttnte/linalg/banded_lu.hpp"
#include "ttnte/linalg/schur.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace ttnte::sweep {

namespace {

constexpr const char* CTX = "ttnte::sweep::ExtrudedSolver";

/// @return Max |x| over a range (the scale of a relative check).
double max_abs(const double* x, int64_t n)
{
  double v = 0.0;
  for (int64_t i = 0; i < n; i++)
    v = std::max(v, std::abs(x[i]));
  return v;
}

} // namespace

// Detection, as for KroneckerSolver, works on the assembled operator rather
// than the geometry. The extruded axis is not known in advance, so every
// (parametric axis s, physical axis e) pairing is tried and the first that
// passes is used. Only 3-D: in 2-D a separable extrusion is an axis-aligned
// rectangle, which KroneckerSolver already handles more cheaply.
ExtrudedSolver::Ptr ExtrudedSolver::try_create(
  const PatchDiscretization& disc, double tol, double max_eigvec_cond)
{
  if (disc.ndim() != 3)
    return nullptr;
  for (int64_t s = 0; s < 3; s++)
    for (int64_t e = 0; e < 3; e++)
      if (auto solver = try_axis(disc, s, e, tol, max_eigvec_cond))
        return solver;
  return nullptr;
}

// Tests L(Omega, sigma) = L_p(Omega_p, sigma) (x) F_s + |Omega_e| M_p (x)
// A_s^{sign} with parametric axis s carried by physical axis e: extract the
// factors from one layer and one line of pairs, then check every pair value
// of the mass, the in-plane gradients, the axial unit directions and nine
// full oblique operators against the factored form, to relative `tol`.
ExtrudedSolver::Ptr ExtrudedSolver::try_axis(const PatchDiscretization& disc,
  int64_t s, int64_t e, double tol, double max_eigvec_cond)
{
  const auto& n = disc.shape();
  const int64_t N = disc.num_dofs();
  const int64_t ns = n[s];
  if (ns < 2)
    return nullptr;
  const int64_t m = N / ns;

  // Digits of the flat (C-order) index, the in-plane index over the two
  // other axes (o0 slower) and its band position (the larger in-plane
  // dimension slowest, which narrows the band)
  const int64_t o0 = s == 0 ? 1 : 0;
  const int64_t o1 = s == 2 ? 1 : 2;
  const int64_t stride[3] = {n[1] * n[2], n[2], 1};
  auto digit = [&](
                 int64_t flat, int64_t i) { return (flat / stride[i]) % n[i]; };
  auto plane = [&](int64_t flat) {
    return digit(flat, o0) * n[o1] + digit(flat, o1);
  };
  const bool swap = n[o1] > n[o0];
  auto position = [&](int64_t a) {
    return swap ? (a % n[o1]) * n[o0] + a / n[o1] : a;
  };

  auto rows = disc.pair_row().contiguous();
  auto cols = disc.pair_col().contiguous();
  auto mass_t = disc.mass_values().contiguous();
  auto grad_t = disc.grad_values().contiguous();
  const int64_t P = rows.size(0);
  const int64_t* r = rows.data_ptr<int64_t>();
  const int64_t* c = cols.data_ptr<int64_t>();
  const double* mass = mass_t.data_ptr<double>();
  const double* grad = grad_t.data_ptr<double>();

  // Reference layer of axis s (mode ref) and reference in-plane dof aref,
  // both at the centre so they are interior functions when the patch has
  // any (nonzero entries to normalize by)
  const int64_t ref = ns / 2;
  const int64_t aref = (n[o0] / 2) * n[o1] + n[o1] / 2;

  // In-plane pairs: the patch pairs with both axial digits at the reference
  // layer. For a partially separable operator every pair value is an
  // in-plane value times a 1-D value along s, and this layer supplies the
  // in-plane pattern and values (to_plane: patch pair -> in-plane pair, -1
  // off the layer; plane_pair: (a, b) -> in-plane pair)
  auto solver = Ptr(new ExtrudedSolver());
  solver->axis_ = s;
  solver->physical_ = e;
  solver->ns_ = ns;
  solver->m_ = m;
  solver->num_dofs_ = N;
  std::vector<int64_t> to_plane(P, -1);
  std::unordered_map<int64_t, int64_t> plane_pair;
  std::vector<int64_t> layer_pair;
  for (int64_t p = 0; p < P; p++) {
    if (digit(r[p], s) != ref || digit(c[p], s) != ref)
      continue;
    const int64_t a = plane(r[p]), b = plane(c[p]);
    to_plane[p] = static_cast<int64_t>(layer_pair.size());
    plane_pair[a * m + b] = to_plane[p];
    layer_pair.push_back(p);
  }
  const int64_t P2 = static_cast<int64_t>(layer_pair.size());
  auto find = [&](int64_t a, int64_t b) {
    auto it = plane_pair.find(a * m + b);
    return it == plane_pair.end() ? int64_t(-1) : it->second;
  };

  // 1-D factors along s from the line of pairs at the reference in-plane
  // dof. Factors are defined only up to a scalar; normalizing F_s(ref, ref)
  // = 1 makes the in-plane factors the reference layer's values themselves
  // (M_p = M on the layer), so no in-plane quadrature is redone. A
  // non-positive mass entry, or a layer without the reference pair, rejects
  // the axis
  auto line = [&](const double* values) {
    std::vector<double> S(ns * ns, 0.0);
    for (int64_t p = 0; p < P; p++)
      if (plane(r[p]) == aref && plane(c[p]) == aref)
        S[digit(r[p], s) * ns + digit(c[p], s)] = values[p];
    return S;
  };
  std::vector<double> F = line(mass);
  const double fref = F[ref * ns + ref];
  const int64_t pref = find(aref, aref);
  if (fref <= 0.0 || pref < 0)
    return nullptr;
  for (auto& x : F)
    x /= fref;
  solver->mass_.resize(P2);
  for (int64_t k = 0; k < P2; k++)
    solver->mass_[k] = mass[layer_pair[k]];
  const double mref = solver->mass_[pref];

  // Check: values(p) = X(in-plane pair of p) * Y(axial digits of p) for
  // every patch pair, relative to the largest value. A pair whose in-plane
  // indices do not overlap within the layer must be zero
  auto separable = [&](const double* values, const std::vector<double>& X,
                     const std::vector<double>& Y) {
    const double scale = max_abs(values, P);
    for (int64_t p = 0; p < P; p++) {
      const int64_t k = find(plane(r[p]), plane(c[p]));
      const double x = k < 0 ? 0.0 : X[k];
      const double expect = x * Y[digit(r[p], s) * ns + digit(c[p], s)];
      if (std::abs(values[p] - expect) > tol * scale)
        return false;
    }
    return true;
  };
  // M = M_p (x) F_s: fails if the cross-section changes along s (tapered),
  // or if s is not a separable axis at all
  if (!separable(mass, solver->mass_, F))
    return nullptr;

  // In-plane gradient components G_i = G_i^p (x) F_s for the two physical
  // components other than e: these hold when the in-plane coordinates do not
  // depend on xi_s (the Jacobian is block diagonal). The component along e
  // is the axial streaming, carried by A_s below
  solver->grad_.assign(3 * P2, 0.0);
  for (int64_t i = 0; i < 3; i++) {
    if (i == e)
      continue;
    std::vector<double> G(P2);
    for (int64_t k = 0; k < P2; k++)
      G[k] = grad[i * P + layer_pair[k]];
    if (!separable(grad + i * P, G, F))
      return nullptr;
    std::copy(G.begin(), G.end(), solver->grad_.begin() + i * P2);
  }

  // Axial factors: the operator of the unit directions +/- e (streaming
  // plus outflow) must be M_p (x) A_s^{+/-}. The side faces see no flow
  // along e (n_e = 0), so only the end caps contribute outflow; A_s is the
  // axial line divided by the in-plane reference mass
  std::array<std::vector<double>, 2> A;
  for (int sign = 0; sign < 2; sign++) {
    auto omega = torch::zeros({1, 3}, torch::kFloat64);
    omega[0][e] = sign == 0 ? 1.0 : -1.0;
    auto values = disc.streaming_values(omega)[0].contiguous();
    const double* v = values.data_ptr<double>();
    A[sign] = line(v);
    for (auto& x : A[sign])
      x /= mref;
    if (!separable(v, solver->mass_, A[sign]))
      return nullptr;
  }

  // Side faces (the four faces not normal to axis s): every quadrature
  // normal must have no component along e, else Omega_e would enter the
  // in-plane outflow and the operator would not separate (a sheared
  // extrusion). Their reference-layer entries are kept to rebuild the
  // in-plane outflow per direction, with weights (Omega . n_q)^+ w_q
  const auto& faces = disc.view().faces;
  for (size_t f = 0; f < faces.size(); f++) {
    if (static_cast<int64_t>(f / 2) == s)
      continue;
    const auto& face = faces[f];
    SideFace side;
    side.num_points = face.num_points;
    side.normal.assign(face.normal, face.normal + 3 * face.num_points);
    side.weights.assign(face.weights, face.weights + face.num_points);
    const double scale = max_abs(face.normal, 3 * face.num_points);
    for (int64_t q = 0; q < face.num_points; q++)
      if (std::abs(face.normal[3 * q + e]) > tol * scale)
        return nullptr;
    for (int64_t k = 0; k < face.num_pairs; k++) {
      const int64_t p2 = to_plane[face.pair_volume_index[k]];
      if (p2 < 0)
        continue;
      side.q.push_back(face.pair_q[k]);
      side.pair.push_back(p2);
      side.value.push_back(face.pair_val[k]);
    }
    solver->faces_.push_back(std::move(side));
  }

  // In-plane pattern in band positions (the larger in-plane dimension
  // slowest, narrowing the band), and the band offsets of every pair
  solver->pair_row_.resize(P2);
  solver->pair_col_.resize(P2);
  int64_t kl = 0, ku = 0;
  for (int64_t k = 0; k < P2; k++) {
    const int64_t i = position(plane(r[layer_pair[k]]));
    const int64_t j = position(plane(c[layer_pair[k]]));
    solver->pair_row_[k] = i;
    solver->pair_col_[k] = j;
    kl = std::max(kl, i - j);
    ku = std::max(ku, j - i);
  }
  solver->kl_ = kl;
  solver->ku_ = ku;
  solver->ldab_ = linalg::banded::factor_ldab(kl, ku);
  solver->band_index_.resize(P2);
  for (int64_t k = 0; k < P2; k++)
    solver->band_index_[k] = linalg::banded::band_offset(
      solver->pair_row_[k], solver->pair_col_[k], kl, ku, solver->ldab_);

  // Flat index of (mode j, band position) for gathers and scatters
  solver->flat_.resize(N);
  for (int64_t flat = 0; flat < N; flat++)
    solver->flat_[digit(flat, s) * m + position(plane(flat))] = flat;

  // Whole-operator check: the separate checks above do not cover the side
  // faces' upwinding, whose outflow set depends on the direction, so the
  // full operator of nine directions (the eight sign patterns of
  // (0.48, 0.6, 0.64), one direction in every octant, and one with
  // Omega_e = 0) is compared with L_p (x) F_s + |Omega_e| M_p (x) A_s
  // assembled from the stored factors
  {
    std::vector<std::array<double, 3>> dirs;
    const double base[3] = {0.48, 0.6, 0.64};
    for (int sx = -1; sx <= 1; sx += 2)
      for (int sy = -1; sy <= 1; sy += 2)
        for (int sz = -1; sz <= 1; sz += 2)
          dirs.push_back({sx * base[0], sy * base[1], sz * base[2]});
    std::array<double, 3> flat_dir = {0.0, 0.0, 0.0};
    flat_dir[(e + 1) % 3] = 0.6;
    flat_dir[(e + 2) % 3] = -0.8;
    dirs.push_back(flat_dir);
    std::vector<double> Sp(P2), clip;
    for (const auto& dir : faces)
      clip.resize(std::max<size_t>(clip.size(), dir.num_points));
    for (const auto& w : dirs) {
      auto omega = torch::tensor({w[0], w[1], w[2]}, torch::kFloat64);
      auto values = disc.streaming_values(omega.view({1, 3}))[0].contiguous();
      const double* v = values.data_ptr<double>();
      solver->assemble_plane(w.data(), 0.0, Sp.data(), clip.data());
      const int sign = w[e] >= 0.0 ? 0 : 1;
      const double mag = std::abs(w[e]);
      const double scale = max_abs(v, P);
      for (int64_t p = 0; p < P; p++) {
        const int64_t k = find(plane(r[p]), plane(c[p]));
        const int64_t jl = digit(r[p], s) * ns + digit(c[p], s);
        const double expect =
          k < 0 ? 0.0 : Sp[k] * F[jl] + mag * solver->mass_[k] * A[sign][jl];
        if (std::abs(v[p] - expect) > tol * scale)
          return nullptr;
      }
    }
  }

  // Every check passed. Per sign of Omega_e: F_s^{-1}, the complex Schur
  // form F_s^{-1} A_s^{+/-} = U R U^* and P = U^* F_s^{-1} (the Schur path),
  // and the eigen-decomposition modes for the diagonal path (empty when
  // they are too ill conditioned; solve() then uses Schur)
  auto Finv = torch::linalg_inv(
    torch::from_blob(F.data(), {ns, ns}, torch::kFloat64).clone())
                .contiguous();
  const double* fi = Finv.data_ptr<double>();
  solver->Finv_.assign(fi, fi + ns * ns);
  for (int sign = 0; sign < 2; sign++) {
    auto T =
      Finv.mm(torch::from_blob(A[sign].data(), {ns, ns}, torch::kFloat64))
        .to(torch::kComplexDouble)
        .contiguous();
    std::vector<cplx> R(ns * ns), U(ns * ns);
    const auto* t =
      reinterpret_cast<const cplx*>(T.data_ptr<c10::complex<double>>());
    std::copy(t, t + ns * ns, R.begin());
    if (!linalg::complex_schur(ns, R.data(), U.data()))
      throw utils::runtime_error(CTX, "Schur decomposition did not converge");
    std::vector<cplx> Pm(ns * ns, 0.0);
    for (int64_t a = 0; a < ns; a++)
      for (int64_t k = 0; k < ns; k++) {
        const cplx u = std::conj(U[k * ns + a]);
        for (int64_t b = 0; b < ns; b++)
          Pm[a * ns + b] += u * fi[k * ns + b];
      }
    solver->P_[sign] = std::move(Pm);
    solver->U_[sign] = std::move(U);
    solver->R_[sign] = std::move(R);
    solver->modes_[sign] = diagonalize(
      Finv.mm(torch::from_blob(A[sign].data(), {ns, ns}, torch::kFloat64)),
      Finv, max_eigvec_cond);
  }
  return solver;
}

ExtrudedSolver::Modes ExtrudedSolver::diagonalize(
  const torch::Tensor& T, const torch::Tensor& Finv, double max_cond)
{
  // Diagonal path: T = V Lambda V^{-1}, so the axial modes decouple. The
  // real T has real eigenvalues and conjugate pairs; with b real a pair's
  // solutions are conjugate, so only a representative of each is solved
  // (x += 2 Re(v w)). Returns no modes (Schur path) if the pairing fails or
  // cond(V) > max_cond (the error grows like cond(V) * eps)
  const int64_t n = T.size(0);
  if (max_cond <= 0.0)
    return {};
  auto [lam_t, vec_t] = torch::linalg_eig(T);
  auto V = vec_t.contiguous().clone();
  auto* lam = reinterpret_cast<cplx*>(lam_t.data_ptr<c10::complex<double>>());
  auto* v = reinterpret_cast<cplx*>(V.data_ptr<c10::complex<double>>());
  double scale = 0.0;
  for (int64_t j = 0; j < n; j++)
    scale = std::max(scale, std::abs(lam[j]));
  const double imag_tol = 1e-10 * std::max(scale, 1.0);

  // Representatives: real eigenvalues (eigenvector made real) and the
  // Im > 0 member of each pair, whose partner column becomes its exact
  // conjugate (so V^{-1}'s rows pair the same way)
  std::vector<int64_t> reps;
  std::vector<uint8_t> real, used(n, 0);
  for (int64_t j = 0; j < n; j++) {
    if (std::abs(lam[j].imag()) > imag_tol)
      continue;
    int64_t big = 0;
    for (int64_t i = 1; i < n; i++)
      if (std::abs(v[i * n + j]) > std::abs(v[big * n + j]))
        big = i;
    const cplx phase = std::conj(v[big * n + j]) / std::abs(v[big * n + j]);
    for (int64_t i = 0; i < n; i++)
      v[i * n + j] = (v[i * n + j] * phase).real();
    lam[j] = lam[j].real();
    used[j] = 1;
    reps.push_back(j);
    real.push_back(1);
  }
  for (int64_t j = 0; j < n; j++) {
    if (used[j] || lam[j].imag() <= 0.0)
      continue;
    int64_t partner = -1;
    for (int64_t k = 0; k < n; k++)
      if (!used[k] && k != j && lam[k].imag() < 0.0 &&
          (partner < 0 || std::abs(lam[k] - std::conj(lam[j])) <
                            std::abs(lam[partner] - std::conj(lam[j]))))
        partner = k;
    if (partner < 0)
      return {};
    for (int64_t i = 0; i < n; i++)
      v[i * n + partner] = std::conj(v[i * n + j]);
    lam[partner] = std::conj(lam[j]);
    used[j] = used[partner] = 1;
    reps.push_back(j);
    real.push_back(0);
  }
  for (int64_t j = 0; j < n; j++)
    if (!used[j])
      return {};
  if (torch::linalg_cond(V).item<double>() > max_cond)
    return {};

  // Q = V^{-1} F_s^{-1} (the modal projection of the right-hand side),
  // kept for the representatives only; a real mode's row is made real
  auto Q = torch::linalg_inv(V).mm(Finv.to(torch::kComplexDouble)).contiguous();
  const auto* q =
    reinterpret_cast<const cplx*>(Q.data_ptr<c10::complex<double>>());
  Modes modes;
  const int64_t k = static_cast<int64_t>(reps.size());
  modes.real = real;
  modes.Q.resize(k * n);
  modes.V.resize(n * k);
  for (int64_t r = 0; r < k; r++) {
    const int64_t j = reps[r];
    modes.lambda.push_back(lam[j]);
    for (int64_t l = 0; l < n; l++) {
      modes.Q[r * n + l] = real[r] ? cplx(q[j * n + l].real()) : q[j * n + l];
      modes.V[l * k + r] = v[l * n + j];
    }
  }
  return modes;
}

void ExtrudedSolver::assemble_plane(
  const double* omega, double sigma, double* values, double* clip) const
{
  const int64_t P2 = static_cast<int64_t>(mass_.size());
  for (int64_t k = 0; k < P2; k++) {
    double v = sigma * mass_[k];
    for (int64_t i = 0; i < 3; i++)
      if (i != physical_)
        v -= omega[i] * grad_[i * P2 + k];
    values[k] = v;
  }
  // Outflow through the side faces: c_q = max(Omega . n_q, 0) w_q
  for (const auto& face : faces_) {
    bool any = false;
    for (int64_t q = 0; q < face.num_points; q++) {
      const double* nq = face.normal.data() + 3 * q;
      const double dot = omega[0] * nq[0] + omega[1] * nq[1] + omega[2] * nq[2];
      clip[q] = dot > 0.0 ? dot * face.weights[q] : 0.0;
      any |= clip[q] != 0.0;
    }
    if (!any)
      continue;
    for (size_t k = 0; k < face.q.size(); k++)
      values[face.pair[k]] += clip[face.q[k]] * face.value[k];
  }
}

// Solves L x = b with L = L_p(Omega_p, sigma) (x) F_s + |Omega_e| M_p (x)
// A_s^{sign}. Working in the layout (axial mode j slowest, in-plane band
// position a fastest), the operator reads
//   (F_s (x) L_p + |Omega_e| A_s (x) M_p) x = b,
// i.e. after multiplying by F_s^{-1} along s,
//   (I (x) L_p + |Omega_e| T (x) M_p) x = (F_s^{-1} (x) I) b,
//   T = F_s^{-1} A_s^{sign}.
// The n_s x n_s factor T couples the layers; the m x m in-plane matrices
// L_p, M_p carry any curvature. Three paths, by what T allows:
//   Omega_e = 0: T drops out, every layer is the same real in-plane solve;
//   diagonal:    T = V Lambda V^{-1} decouples the layers into independent
//                shifted in-plane solves (L_p + |Omega_e| lambda M_p);
//   Schur:       T = U R U^* makes them block upper triangular, solved last
//                mode first.
// Every in-plane system is shifted by |Omega_e| times an eigenvalue of T,
// whose real part is >= 0, so it stays as well posed as L_p itself.
bool ExtrudedSolver::solve(
  const double* omega, double sigma, double* b, Workspace& work) const
{
  const int64_t ns = ns_, m = m_;
  const int64_t P2 = static_cast<int64_t>(mass_.size());
  const int64_t size = m * ldab_;
  int64_t qmax = 0;
  for (const auto& face : faces_)
    qmax = std::max(qmax, face.num_points);
  work.values.resize(P2);
  work.clip.resize(qmax);
  work.layers.resize(2 * ns * m);
  work.pivots.resize(m);
  double* values = work.values.data();
  double* layers = work.layers.data();
  int64_t* piv = work.pivots.data();
  // The in-plane operator L_p = sigma M_p - sum_{i != e} Omega_i G_i^p plus
  // the side-face outflow for this direction, as in-plane pair values. It
  // is the only direction-dependent assembly; every other factor is stored
  assemble_plane(omega, sigma, values, work.clip.data());

  // Gather b into (axial index l, band position a): row l of `layers` is
  // layer l of the patch, in the in-plane band ordering. The second half of
  // `layers` is scratch for the result
  for (int64_t k = 0; k < ns * m; k++)
    layers[k] = b[flat_[k]];

  const double mag = std::abs(omega[physical_]);
  if (mag == 0.0) {
    // No flow along e: L = L_p (x) F_s, so x = (F_s^{-1} (x) L_p^{-1}) b:
    // apply F_s^{-1} along the axis (mixing layers), then one real in-plane
    // factorization, reused for all n_s layers
    double* g = layers + ns * m;
    std::fill(g, g + ns * m, 0.0);
    for (int64_t j = 0; j < ns; j++)
      for (int64_t l = 0; l < ns; l++) {
        const double f = Finv_[j * ns + l];
        if (f == 0.0)
          continue;
        for (int64_t a = 0; a < m; a++)
          g[j * m + a] += f * layers[l * m + a];
      }
    work.real_band.assign(size, 0.0);
    double* L = work.real_band.data();
    for (int64_t k = 0; k < P2; k++)
      L[band_index_[k]] = values[k];
    if (linalg::banded::lu_factor(L, ldab_, m, kl_, ku_, piv) != 0)
      throw utils::runtime_error(CTX, "Singular in-plane system");
    for (int64_t j = 0; j < ns; j++)
      linalg::banded::lu_solve(L, ldab_, m, kl_, ku_, piv, g + j * m);
    for (int64_t k = 0; k < ns * m; k++)
      b[flat_[k]] = g[k];
    return true;
  }

  // The sign of Omega_e selects A_s^+ (outflow at the upper end of the axis)
  // or A_s^-, and with it the stored factorizations
  const int sign = omega[physical_] >= 0.0 ? 0 : 1;
  const Modes& md = modes_[sign];
  if (!md.lambda.empty()) {
    // Diagonal path. With x = (V (x) I) w and Q = V^{-1} F_s^{-1}, each mode
    // r is independent:
    //   (L_p + |Omega_e| lambda_r M_p) w_r = (Q b)_r = sum_l Q_rl b_l,
    // and x_l = sum_r V_lr w_r. Only representatives are stored: a real
    // lambda gives a real system (real Q row and eigenvector); a conjugate
    // pair's two solutions are conjugates of each other (b, L_p, M_p real),
    // so one complex solve covers both, contributing
    //   v w + conj(v w) = 2 Re(v w)
    // to x. One band factorization per representative, about n_s / 2
    // complex ones in all
    const int64_t k = static_cast<int64_t>(md.lambda.size());
    double* x = layers + ns * m;
    std::fill(x, x + ns * m, 0.0);
    work.rhs.resize(m);
    work.f.resize(m);
    double* rr = work.rhs.data();
    cplx* rc = work.f.data();
    for (int64_t r = 0; r < k; r++) {
      const cplx* q = md.Q.data() + r * ns;
      if (md.real[r]) {
        // Real mode: right-hand side sum_l Q_rl b_l, the real shifted band
        // system, then x_l += V_lr w_r
        std::fill(rr, rr + m, 0.0);
        for (int64_t l = 0; l < ns; l++) {
          const double c = q[l].real();
          for (int64_t a = 0; a < m; a++)
            rr[a] += c * layers[l * m + a];
        }
        const double shift = mag * md.lambda[r].real();
        work.real_band.assign(size, 0.0);
        double* L = work.real_band.data();
        for (int64_t p = 0; p < P2; p++)
          L[band_index_[p]] = values[p] + shift * mass_[p];
        if (linalg::banded::lu_factor(L, ldab_, m, kl_, ku_, piv) != 0)
          throw utils::runtime_error(CTX, "Singular in-plane system");
        linalg::banded::lu_solve(L, ldab_, m, kl_, ku_, piv, rr);
        for (int64_t l = 0; l < ns; l++) {
          const double vl = md.V[l * k + r].real();
          for (int64_t a = 0; a < m; a++)
            x[l * m + a] += vl * rr[a];
        }
        continue;
      }
      // Conjugate pair (representative with Im lambda > 0): the same in
      // complex arithmetic, then x_l += 2 Re(V_lr w_r)
      std::fill(rc, rc + m, cplx(0.0));
      for (int64_t l = 0; l < ns; l++) {
        const cplx c = q[l];
        for (int64_t a = 0; a < m; a++)
          rc[a] += c * layers[l * m + a];
      }
      const cplx shift = mag * md.lambda[r];
      work.band.assign(size, cplx(0.0));
      cplx* B = work.band.data();
      for (int64_t p = 0; p < P2; p++)
        B[band_index_[p]] = values[p] + shift * mass_[p];
      if (linalg::banded::lu_factor(B, ldab_, m, kl_, ku_, piv) != 0)
        throw utils::runtime_error(CTX, "Singular in-plane system");
      linalg::banded::lu_solve(B, ldab_, m, kl_, ku_, piv, rc);
      for (int64_t l = 0; l < ns; l++) {
        const cplx vl = md.V[l * k + r];
        // Re(v w) = Re v Re w - Im v Im w
        for (int64_t a = 0; a < m; a++)
          x[l * m + a] +=
            2.0 * (vl.real() * rc[a].real() - vl.imag() * rc[a].imag());
      }
    }
    for (int64_t a = 0; a < ns * m; a++)
      b[flat_[a]] = x[a];
    return true;
  }

  // Schur path, when the axial eigenvectors are too ill conditioned for the
  // diagonal path (cond(V) > max_eigvec_cond). With T = U R U^*,
  // x = (U (x) I) w and P = U^* F_s^{-1}, the system becomes
  //   (I (x) L_p + |Omega_e| R (x) M_p) w = (P (x) I) b =: f,
  // block upper triangular since R is: block j couples only to later
  // modes k > j. Backward stable (U unitary) however non-normal T is, at
  // the cost of n_s complex band factorizations and the coupling products
  const cplx* Pm = P_[sign].data();
  const cplx* U = U_[sign].data();
  const cplx* R = R_[sign].data();

  // f = (P along s) b: f_j = sum_l P_jl b_l, an n_s x n_s mode product;
  // overwritten mode by mode with w
  work.f.assign(ns * m, cplx(0.0));
  cplx* f = work.f.data();
  for (int64_t j = 0; j < ns; j++)
    for (int64_t l = 0; l < ns; l++) {
      const cplx p = Pm[j * ns + l];
      for (int64_t a = 0; a < m; a++)
        f[j * m + a] += p * layers[l * m + a];
    }

  // Block back substitution over the modes, last first:
  //   (L_p + |Omega_e| R_jj M_p) w_j = f_j - |Omega_e| M_p sum_{k>j} R_jk w_k,
  // every w_k on the right already solved. R_jj is an eigenvalue of T
  // (Re >= 0), so each shifted in-plane system is nonsingular
  work.t.resize(m);
  work.band.resize(size);
  cplx* t = work.t.data();
  cplx* band = work.band.data();
  for (int64_t j = ns - 1; j >= 0; j--) {
    cplx* fj = f + j * m;
    if (j < ns - 1) {
      // f_j -= |Omega_e| M_p sum_{k>j} R_jk w_k: first t = sum_k R_jk w_k,
      // then M_p t through the in-plane pair list (row, column, value)
      std::fill(t, t + m, cplx(0.0));
      for (int64_t k = j + 1; k < ns; k++) {
        const cplx rjk = R[j * ns + k];
        if (rjk == cplx(0.0))
          continue;
        const cplx* wk = f + k * m;
        for (int64_t a = 0; a < m; a++)
          t[a] += rjk * wk[a];
      }
      for (int64_t k = 0; k < P2; k++)
        fj[pair_row_[k]] -= mag * mass_[k] * t[pair_col_[k]];
    }
    // (L_p + |Omega_e| R_jj M_p) w_j = f_j
    const cplx shift = mag * R[j * ns + j];
    std::fill(band, band + size, cplx(0.0));
    for (int64_t k = 0; k < P2; k++)
      band[band_index_[k]] = values[k] + shift * mass_[k];
    if (linalg::banded::lu_factor(band, ldab_, m, kl_, ku_, piv) != 0)
      throw utils::runtime_error(CTX, "Singular in-plane system");
    linalg::banded::lu_solve(band, ldab_, m, kl_, ku_, piv, fj);
  }

  // x = Re((U along s) w): x_l = sum_j U_lj w_j. The exact x is real (L
  // and b are), so the imaginary part is rounding and dropped. Then scatter
  // back to the patch's flat order
  for (int64_t l = 0; l < ns; l++) {
    double* xl = layers + l * m;
    std::fill(xl, xl + m, 0.0);
    for (int64_t j = 0; j < ns; j++) {
      const cplx u = U[l * ns + j];
      const cplx* wj = f + j * m;
      for (int64_t a = 0; a < m; a++)
        xl[a] += u.real() * wj[a].real() - u.imag() * wj[a].imag();
    }
  }
  for (int64_t k = 0; k < ns * m; k++)
    b[flat_[k]] = layers[k];
  return true;
}

} // namespace ttnte::sweep
