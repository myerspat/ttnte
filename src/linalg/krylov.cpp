#include "ttnte/linalg/krylov.hpp"
#include "ttnte/utils/exception.hpp"
#include <cmath>

namespace ttnte::linalg::krylov {

namespace {

double dot(const double* a, const double* b, size_t n)
{
  double s = 0.0;
  for (size_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

} // namespace

KrylovResult gmres(const LinearOperator& apply, const std::vector<double>& b,
  std::vector<double>& x, int64_t restart, double tol, int64_t max_applies)
{
  const size_t n = b.size();
  if (x.size() != n) {
    throw utils::runtime_error("ttnte::linalg::krylov::gmres",
      "The initial guess and right-hand side lengths differ");
  }
  if (restart < 1) {
    throw utils::runtime_error(
      "ttnte::linalg::krylov::gmres", "`restart` must be at least 1");
  }

  KrylovResult result;
  const double bnorm = std::sqrt(dot(b.data(), b.data(), n));
  if (bnorm == 0.0) {
    std::fill(x.begin(), x.end(), 0.0);
    result.converged = true;
    return result;
  }

  const int64_t m = restart;
  std::vector<double> V((m + 1) * n), H((m + 1) * m, 0.0);
  std::vector<double> cs(m), sn(m), g(m + 1), w(n), r(n);

  while (true) {
    // True residual (only if the budget allows another application; the
    // reported residual is then the last Arnoldi estimate)
    if (result.num_applies > 0 && result.num_applies >= max_applies) {
      return result;
    }
    apply(x.data(), w.data());
    result.num_applies++;
    for (size_t i = 0; i < n; i++)
      r[i] = b[i] - w[i];
    double beta = std::sqrt(dot(r.data(), r.data(), n));
    result.relative_residual = beta / bnorm;
    if (result.relative_residual <= tol) {
      result.converged = true;
      return result;
    }
    if (result.num_applies >= max_applies) {
      return result;
    }

    for (size_t i = 0; i < n; i++)
      V[i] = r[i] / beta;
    std::fill(g.begin(), g.end(), 0.0);
    g[0] = beta;

    int64_t k = 0;
    bool estimate_converged = false;
    for (; k < m && result.num_applies < max_applies; k++) {
      double* vk1 = V.data() + (k + 1) * n;
      apply(V.data() + k * n, vk1);
      result.num_applies++;

      // Modified Gram-Schmidt
      for (int64_t j = 0; j <= k; j++) {
        const double* vj = V.data() + j * n;
        const double h = dot(vk1, vj, n);
        H[j * m + k] = h;
        for (size_t i = 0; i < n; i++)
          vk1[i] -= h * vj[i];
      }
      const double hk1 = std::sqrt(dot(vk1, vk1, n));
      H[(k + 1) * m + k] = hk1;
      if (hk1 > 0.0) {
        for (size_t i = 0; i < n; i++)
          vk1[i] /= hk1;
      }

      // Apply previous rotations, then form the new one
      for (int64_t j = 0; j < k; j++) {
        const double a = H[j * m + k], c = H[(j + 1) * m + k];
        H[j * m + k] = cs[j] * a + sn[j] * c;
        H[(j + 1) * m + k] = -sn[j] * a + cs[j] * c;
      }
      const double a = H[k * m + k], c = H[(k + 1) * m + k];
      const double rho = std::hypot(a, c);
      cs[k] = rho > 0.0 ? a / rho : 1.0;
      sn[k] = rho > 0.0 ? c / rho : 0.0;
      H[k * m + k] = rho;
      H[(k + 1) * m + k] = 0.0;
      g[k + 1] = -sn[k] * g[k];
      g[k] = cs[k] * g[k];

      result.relative_residual = std::abs(g[k + 1]) / bnorm;
      if (result.relative_residual <= tol || hk1 == 0.0) {
        estimate_converged = true;
        k++;
        break;
      }
    }

    // Solve the upper-triangular system and update x
    std::vector<double> y(k, 0.0);
    for (int64_t i = k - 1; i >= 0; i--) {
      double s = g[i];
      for (int64_t j = i + 1; j < k; j++)
        s -= H[i * m + j] * y[j];
      y[i] = s / H[i * m + i];
    }
    for (int64_t j = 0; j < k; j++) {
      const double* vj = V.data() + j * n;
      for (size_t i = 0; i < n; i++)
        x[i] += y[j] * vj[i];
    }
    // Trust the Arnoldi estimate on convergence (saves an application;
    // callers that need the true residual can check it themselves)
    if (estimate_converged) {
      result.converged = true;
      return result;
    }
  }
}

} // namespace ttnte::linalg::krylov
