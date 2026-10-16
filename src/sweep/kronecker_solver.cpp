#include "ttnte/sweep/kronecker_solver.hpp"
#include "ttnte/linalg/schur.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>

namespace ttnte::sweep {

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

  // Multi-indices of every pair, and the reference index of each dimension
  std::vector<int64_t> stride(d, 1);
  for (int64_t i = d - 2; i >= 0; i--)
    stride[i] = stride[i + 1] * shape[i + 1];
  auto digit = [&](int64_t flat, int64_t i) {
    return (flat / stride[i]) % shape[i];
  };
  std::vector<int64_t> ref(d);
  for (int64_t i = 0; i < d; i++)
    ref[i] = shape[i] / 2;
  // Pairs that vary only along dimension i (the others at the reference)
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

  // Mass factors: F_0 = S_0 / s^(d-1), F_i = S_i, s = M(ref, ref)
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
  for (int64_t p = 0; p < P; p++)
    if (std::abs(m[p] - product(p, -1)) > tol * mmax)
      return nullptr;

  auto solver = Ptr(new KroneckerSolver());
  solver->ndim_ = d;
  solver->n_ = shape;
  solver->num_dofs_ = N;
  for (int64_t i = 0; i < d; i++) {
    const int64_t n = shape[i];
    // F_i^{-1} (small dense, by Gauss-Jordan on a copy)
    auto Finv = torch::linalg_inv(
      torch::from_blob(F[i].data(), {n, n}, torch::kFloat64).clone());
    double fref = 1.0;
    for (int64_t k = 0; k < d; k++)
      if (k != i)
        fref *= F[k][ref[k] * shape[k] + ref[k]];
    for (int sign = 0; sign < 2; sign++) {
      auto omega = torch::zeros({1, d}, torch::kFloat64);
      omega[0][i] = sign == 0 ? 1.0 : -1.0;
      auto values = disc.streaming_values(omega)[0].contiguous();
      const double* v = values.data_ptr<double>();
      // Check the Kronecker structure A_i (x) F_others
      std::vector<double> A = slice(v, i);
      for (auto& x : A)
        x /= fref;
      double vmax = 0.0;
      for (int64_t p = 0; p < P; p++)
        vmax = std::max(vmax, std::abs(v[p]));
      for (int64_t p = 0; p < P; p++) {
        const double expect =
          A[digit(r[p], i) * n + digit(c[p], i)] * product(p, i);
        if (std::abs(v[p] - expect) > tol * vmax)
          return nullptr;
      }
      auto T = Finv.mm(torch::from_blob(A.data(), {n, n}, torch::kFloat64))
                 .to(torch::kComplexDouble)
                 .contiguous();
      std::vector<cplx> R(n * n), U(n * n);
      const auto* t =
        reinterpret_cast<const cplx*>(T.data_ptr<c10::complex<double>>());
      std::copy(t, t + n * n, R.begin());
      if (!linalg::complex_schur(n, R.data(), U.data())) {
        throw utils::runtime_error("ttnte::sweep::KroneckerSolver::try_create",
          "Schur decomposition did not converge");
      }
      // P = U^* F^{-1}
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

void KroneckerSolver::mode_product(
  const cplx* B, int64_t dim, const cplx* in, cplx* out) const
{
  const int64_t n = n_[dim];
  int64_t outer = 1, inner = 1;
  for (int64_t k = 0; k < dim; k++)
    outer *= n_[k];
  for (int64_t k = dim + 1; k < ndim_; k++)
    inner *= n_[k];
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

void KroneckerSolver::solve(
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
  std::vector<int64_t> stride(ndim_, 1);
  for (int64_t i = ndim_ - 2; i >= 0; i--)
    stride[i] = stride[i + 1] * n_[i + 1];
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
}

} // namespace ttnte::sweep
