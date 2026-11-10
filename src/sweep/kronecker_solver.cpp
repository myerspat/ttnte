#include "ttnte/sweep/kronecker_solver.hpp"
#include "ttnte/linalg/schur.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>

namespace ttnte::sweep {

namespace {

/// @return Row-major strides of a flat index over `shape`.
std::vector<int64_t> row_major_strides(const std::vector<int64_t>& shape)
{
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--)
    stride[i] = stride[i + 1] * shape[i + 1];
  return stride;
}

} // namespace

// Detection works on the assembled operator, not the geometry: anything whose
// mass and unit-direction streaming are Kronecker products to `tol` passes,
// whatever mapping produced it. The patch's operators come as lists of pair
// values (row a, column b of every pair of overlapping basis functions);
// with a row-major flat index each a has digits (a_1, ..., a_d), one per
// parametric axis. If L = sigma (F_1 x ... x F_d) + sum_i |Omega_i|
// F_1 x ... x A_i^{sign} x ... x F_d, then every pair value is a product of
// 1-D entries, M_ab = prod_k (F_k)_{a_k b_k}, which is what is checked.
KroneckerSolver::Ptr KroneckerSolver::try_create(
  const PatchDiscretization& disc, double tol)
{
  const int64_t d = disc.ndim();
  const auto& shape = disc.shape();
  const int64_t N = disc.num_dofs();
  auto rows = disc.pair_row().contiguous();
  auto cols = disc.pair_col().contiguous();
  const int64_t P = rows.size(0);
  const int64_t* r = rows.data_ptr<int64_t>();
  const int64_t* c = cols.data_ptr<int64_t>();

  // Multi-indices of every pair, and the reference index r of each
  // dimension (the centre, so the reference functions are interior ones
  // when the patch has any: their 1-D entries are nonzero)
  const auto stride = row_major_strides(shape);
  auto digit = [&](int64_t flat, int64_t i) {
    return (flat / stride[i]) % shape[i];
  };
  std::vector<int64_t> ref(d);
  for (int64_t i = 0; i < d; i++)
    ref[i] = shape[i] / 2;
  // Pairs that vary only along dimension i: both row and column digits equal
  // the reference in every other dimension. For a Kronecker product those
  // pairs form the n_i x n_i slice S_i = F_i * prod_{k != i} (F_k)_{r_k r_k},
  // i.e. the 1-D factor up to a known scalar
  auto on_line = [&](int64_t p, int64_t i) {
    for (int64_t k = 0; k < d; k++)
      if (k != i && (digit(r[p], k) != ref[k] || digit(c[p], k) != ref[k]))
        return false;
    return true;
  };
  auto slice = [&](const double* values, int64_t i) {
    std::vector<double> S(shape[i] * shape[i], 0.0);
    for (int64_t p = 0; p < P; p++)
      if (on_line(p, i))
        S[digit(r[p], i) * shape[i] + digit(c[p], i)] = values[p];
    return S;
  };

  // Mass factors. Kronecker factors are only defined up to scalars
  // (c F_1 x F_2 / c = F_1 x F_2), so any consistent choice will do: with
  // s = M(r, r) = prod_k (F_k)_{r_k r_k}, the product of the d slices is the
  // true product times s^(d-1), so take F_i = S_i for i > 0 and divide the
  // first by s^(d-1). s <= 0 (not a mass) rejects the patch
  auto mass = disc.mass_values().contiguous();
  const double* m = mass.data_ptr<double>();
  std::vector<std::vector<double>> F(d);
  double s_ref = 0.0;
  for (int64_t p = 0; p < P; p++)
    if (on_line(p, 0) && digit(r[p], 0) == ref[0] && digit(c[p], 0) == ref[0])
      s_ref = m[p];
  if (s_ref <= 0.0)
    return nullptr;
  for (int64_t i = 0; i < d; i++)
    F[i] = slice(m, i);
  for (auto& x : F[0])
    x /= std::pow(s_ref, static_cast<double>(d - 1));

  // prod_k (F_k)_{a_k b_k} for pair p, leaving out dimension `skip` (-1:
  // none)
  auto product = [&](int64_t p, int64_t skip) {
    double v = 1.0;
    for (int64_t k = 0; k < d; k++)
      if (k != skip)
        v *= F[k][digit(r[p], k) * shape[k] + digit(c[p], k)];
    return v;
  };
  double mmax = 0.0;
  for (int64_t p = 0; p < P; p++)
    mmax = std::max(mmax, std::abs(m[p]));
  // Check every mass pair against the product of its 1-D entries, relative
  // to the largest entry. A curved or skewed patch (Jacobian not diagonal)
  // fails here
  for (int64_t p = 0; p < P; p++)
    if (std::abs(m[p] - product(p, -1)) > tol * mmax)
      return nullptr;

  // Streaming factors, per dimension and sign: the operator of the unit
  // direction +e_i or -e_i (streaming plus its outflow faces) must be
  // A_i^{+/-} along i times the mass factors of the other dimensions. A box
  // whose parametric axes are not the physical axes in order (rotated) fails
  // here, since e_i then streams along another parametric axis; a reversed
  // axis (x_i' < 0) passes, its swapped outflow end read from the data.
  // Unit directions determine every direction: within an octant L is affine
  // in (sigma, |Omega_1|, ..., |Omega_d|) when every face normal is +/- e_k,
  // and a patch without axis-aligned faces has already failed a check

  std::vector<std::array<std::vector<double>, 2>> A(d);
  for (int64_t i = 0; i < d; i++) {
    const int64_t n = shape[i];
    double fref = 1.0;
    for (int64_t k = 0; k < d; k++)
      if (k != i)
        fref *= F[k][ref[k] * shape[k] + ref[k]];
    for (int sign = 0; sign < 2; sign++) {
      auto omega = torch::zeros({1, d}, torch::kFloat64);
      omega[0][i] = sign == 0 ? 1.0 : -1.0;
      auto values = disc.streaming_values(omega)[0].contiguous();
      const double* v = values.data_ptr<double>();
      // The slice along i divided by the other dimensions' reference
      // entries is A_i; then check every pair against
      // (A_i)_{a_i b_i} prod_{k != i} (F_k)_{a_k b_k}
      std::vector<double> Ai = slice(v, i);
      for (auto& x : Ai)
        x /= fref;
      double vmax = 0.0;
      for (int64_t p = 0; p < P; p++)
        vmax = std::max(vmax, std::abs(v[p]));
      for (int64_t p = 0; p < P; p++) {
        const double expect =
          Ai[digit(r[p], i) * n + digit(c[p], i)] * product(p, i);
        if (std::abs(v[p] - expect) > tol * vmax)
          return nullptr;
      }
      A[i][sign] = std::move(Ai);
    }
  }
  // Every check passed: factor the 1-D operators (F_i^{-1}, the complex
  // Schur forms of F_i^{-1} A_i^{+/-}, P = U^* F_i^{-1})
  return from_factors(shape, F, A);
}

// The factorization shared by every direction and group of the patch. With
// L = (F_1 x ... x F_d)(sigma I + sum_i |Omega_i| I_{<i} x T_i x I_{>i}),
// T_i = F_i^{-1} A_i^{sign}, and the Schur forms T_i = U_i R_i U_i^*, one
// unitary U_1 x ... x U_d triangularizes every term, so a solve is
// x = Re[(U_1 x ... x U_d) R^{-1} (P_1 x ... x P_d) b], P_i = U_i^* F_i^{-1},
// with R = sigma I + sum_i |Omega_i| I_{<i} x R_i x I_{>i} upper triangular
// in the flat order. Only the signs of Omega pick among the stored
// factors; sigma and |Omega_i| enter R's entries at solve time. Called by
// try_create (factors read from the patch) and parametric_twin.
KroneckerSolver::Ptr KroneckerSolver::from_factors(
  const std::vector<int64_t>& shape, const std::vector<std::vector<double>>& F,
  const std::vector<std::array<std::vector<double>, 2>>& A)
{
  const int64_t d = static_cast<int64_t>(shape.size());
  if (d < 1 || d > 3 || static_cast<int64_t>(F.size()) != d ||
      static_cast<int64_t>(A.size()) != d) {
    throw utils::runtime_error("ttnte::sweep::KroneckerSolver::from_factors",
      "Expected 1 to 3 dimensions, with one F and one pair A per dimension");
  }
  for (int64_t i = 0; i < d; i++) {
    const size_t nn = static_cast<size_t>(shape[i] * shape[i]);
    if (F[i].size() != nn || A[i][0].size() != nn || A[i][1].size() != nn)
      throw utils::runtime_error("ttnte::sweep::KroneckerSolver::from_factors",
        "Factor sizes do not match the shape");
  }
  // Shape, row-major strides (dimension 0 slowest) and size of the flat index
  auto solver = Ptr(new KroneckerSolver());
  solver->ndim_ = d;
  solver->n_ = shape;
  solver->stride_ = row_major_strides(shape);
  solver->num_dofs_ = 1;
  for (int64_t n : shape)
    solver->num_dofs_ *= n;
  // Per dimension: the n_i x n_i factors, 2d small factorizations in all
  // (O(n_i^3) each, done once per patch)
  for (int64_t i = 0; i < d; i++) {
    const int64_t n = shape[i];
    // F_i is the 1-D mass, symmetric positive definite, so the inverse is
    // well defined; it is formed explicitly because n_i is small and
    // F_i^{-1} enters both T_i and P_i
    auto Finv =
      torch::linalg_inv(torch::from_blob(const_cast<double*>(F[i].data()),
                          {n, n}, torch::kFloat64)
                          .clone())
        .contiguous();
    // sign 0: flow along +axis i (outflow at the upper end, A_i^+); sign 1:
    // along -axis i (A_i^-)
    for (int sign = 0; sign < 2; sign++) {
      // T_i = F_i^{-1} A_i, in complex arithmetic: it is real but
      // non-normal with complex-conjugate eigenvalues, so its triangular
      // (complex Schur) form is complex. Schur rather than an
      // eigendecomposition: U is unitary, so the solve is backward stable
      // however ill conditioned T's eigenvectors are (their condition number
      // grows quickly with n_i, and a diagonal solve would pay its d-th power)
      auto T = Finv
                 .mm(torch::from_blob(const_cast<double*>(A[i][sign].data()),
                   {n, n}, torch::kFloat64))
                 .to(torch::kComplexDouble)
                 .contiguous();
      std::vector<cplx> R(n * n), U(n * n);
      const auto* t =
        reinterpret_cast<const cplx*>(T.data_ptr<c10::complex<double>>());
      std::copy(t, t + n * n, R.begin());
      // T = U R U^* (LAPACK zgees), R upper triangular with T's eigenvalues
      // on its diagonal; their real parts are >= 0 (A_i + A_i^T is positive
      // semidefinite), so the pivots of R are at least sigma in modulus
      if (!linalg::complex_schur(n, R.data(), U.data())) {
        throw utils::runtime_error(
          "ttnte::sweep::KroneckerSolver::from_factors",
          "Schur decomposition did not converge");
      }
      // P = U^* F^{-1}: the right-hand side's transform into the Schur basis
      // (the mass inverse folded in, so a solve is d mode products by P,
      // the triangular solve, and d mode products by U)
      std::vector<cplx> Pm(n * n, 0.0);
      const double* fi = Finv.data_ptr<double>();
      for (int64_t a = 0; a < n; a++)
        for (int64_t k = 0; k < n; k++) {
          const cplx u = std::conj(U[k * n + a]);
          for (int64_t b = 0; b < n; b++)
            Pm[a * n + b] += u * fi[k * n + b];
        }
      solver->P_[i][sign] = std::move(Pm);
      solver->U_[i][sign] = std::move(U);
      solver->R_[i][sign] = std::move(R);
    }
  }
  return solver;
}

KroneckerSolver::Ptr KroneckerSolver::parametric_twin(
  const PatchDiscretization& disc, torch::Tensor& metric, double& measure)
{
  const auto& bases = disc.dim_basis();
  const int64_t d = static_cast<int64_t>(bases.size());
  std::vector<int64_t> shape;
  std::vector<std::vector<double>> F(d);
  std::vector<std::array<std::vector<double>, 2>> A(d);
  for (int64_t i = 0; i < d; i++) {
    const auto& b = bases[i];
    const int64_t n = b.num_functions;
    const int64_t nq = b.first.size(0);
    const int64_t p1 = b.values.size(1);
    const int64_t* first = b.first.data_ptr<int64_t>();
    const double* N = b.values.data_ptr<double>();
    const double* D = b.derivs.data_ptr<double>();
    const double* w = b.weights.data_ptr<double>();
    std::vector<double> M(n * n, 0.0), K(n * n, 0.0);
    for (int64_t q = 0; q < nq; q++)
      for (int64_t s = 0; s < p1; s++)
        for (int64_t t = 0; t < p1; t++) {
          const int64_t a = first[q] + s, c = first[q] + t;
          M[a * n + c] += w[q] * N[q * p1 + s] * N[q * p1 + t];
          // Streaming tests the derivative: (d R_a, R_c)
          K[a * n + c] += w[q] * D[q * p1 + s] * N[q * p1 + t];
        }
    // Unit velocity along +axis: -K + outflow at the upper end; along
    // -axis: K + outflow at the lower end (B-splines interpolate the ends)
    std::vector<double> plus(n * n), minus(n * n);
    for (int64_t x = 0; x < n * n; x++) {
      plus[x] = -K[x];
      minus[x] = K[x];
    }
    plus[(n - 1) * n + (n - 1)] += 1.0;
    minus[0] += 1.0;
    shape.push_back(n);
    F[i] = std::move(M);
    A[i] = {std::move(plus), std::move(minus)};
  }
  // Patch averages of |det J| J^{-1} and |det J| over the parametric box
  auto measure_q = disc.quad_measure();
  // Parametric quadrature weights: the outer product of the 1-D weights
  auto flat = torch::ones({1}, torch::kFloat64);
  for (int64_t i = 0; i < d; i++) {
    std::vector<int64_t> v(d, 1);
    v[i] = bases[i].weights.size(0);
    flat = flat * bases[i].weights.reshape(v);
  }
  const auto param = flat.reshape({-1});
  const double volume = param.sum().item<double>();
  metric = (measure_q.view({-1, 1, 1}) * disc.quad_inverse_jacobian())
             .sum(0)
             .div(volume)
             .contiguous();
  measure = measure_q.sum().item<double>() / volume;
  return from_factors(shape, F, A);
}

void KroneckerSolver::mode_product(
  const cplx* B, int64_t dim, const cplx* in, cplx* out) const
{
  const int64_t n = n_[dim];
  const int64_t inner = stride_[dim];
  const int64_t outer = num_dofs_ / (n * inner);
  for (int64_t o = 0; o < outer; o++) {
    const cplx* src = in + o * n * inner;
    cplx* dst = out + o * n * inner;
    for (int64_t a = 0; a < n; a++) {
      cplx* row = dst + a * inner;
      std::fill(row, row + inner, cplx(0.0));
      for (int64_t b = 0; b < n; b++) {
        const cplx coef = B[a * n + b];
        if (coef == cplx(0.0))
          continue;
        const cplx* col = src + b * inner;
        for (int64_t t = 0; t < inner; t++)
          row[t] += coef * col[t];
      }
    }
  }
}

bool KroneckerSolver::solve(
  const double* omega, double sigma, double* b, Workspace& work) const
{
  const int64_t N = num_dofs_;
  work.f.resize(N);
  work.g.resize(N);
  cplx* f = work.f.data();
  cplx* g = work.g.data();
  int sign[3] = {0, 0, 0};
  double mag[3] = {0.0, 0.0, 0.0};
  for (int64_t i = 0; i < ndim_; i++) {
    sign[i] = omega[i] >= 0.0 ? 0 : 1;
    mag[i] = std::abs(omega[i]);
  }

  // f = (P_1 x ... x P_d) b
  for (int64_t k = 0; k < N; k++)
    f[k] = b[k];
  for (int64_t i = 0; i < ndim_; i++) {
    mode_product(P_[i][sign[i]].data(), i, f, g);
    std::swap(f, g);
  }

  // (sigma + sum_i |Omega_i| R_i) w = f, R_i upper triangular: descending
  // flat order sees every larger index of each dimension first
  const auto& stride = stride_;
  const cplx* R[3] = {nullptr, nullptr, nullptr};
  for (int64_t i = 0; i < ndim_; i++)
    R[i] = R_[i][sign[i]].data();
  for (int64_t k = N - 1; k >= 0; k--) {
    cplx rhs = f[k];
    cplx diag = sigma;
    for (int64_t i = 0; i < ndim_; i++) {
      if (mag[i] == 0.0)
        continue;
      const int64_t n = n_[i];
      const int64_t a = (k / stride[i]) % n;
      const cplx* row = R[i] + a * n;
      cplx sum = 0.0;
      for (int64_t j = a + 1; j < n; j++)
        sum += row[j] * f[k + (j - a) * stride[i]];
      rhs -= mag[i] * sum;
      diag += mag[i] * row[a];
    }
    f[k] = rhs / diag;
  }

  // x = Re((U_1 x ... x U_d) w)
  for (int64_t i = 0; i < ndim_; i++) {
    mode_product(U_[i][sign[i]].data(), i, f, g);
    std::swap(f, g);
  }
  for (int64_t k = 0; k < N; k++)
    b[k] = f[k].real();
  return true;
}

} // namespace ttnte::sweep
