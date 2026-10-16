#include "ttnte/linalg/schur.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ttnte::linalg {

namespace {

using cplx = std::complex<double>;

/// Rotation G = [[c, s], [-conj(s), c]] (c real) with G [a; b] = [r; 0].
void givens(cplx a, cplx b, double& c, cplx& s)
{
  const double na = std::abs(a), nb = std::abs(b);
  if (nb == 0.0) {
    c = 1.0;
    s = 0.0;
  } else if (na == 0.0) {
    c = 0.0;
    s = std::conj(b) / nb;
  } else {
    const double r = std::hypot(na, nb);
    c = na / r;
    s = (a / na) * std::conj(b) / r;
  }
}

} // namespace

bool complex_schur(int64_t n, cplx* a, cplx* q, int64_t max_iterations)
{
  auto A = [a, n](int64_t i, int64_t j) -> cplx& { return a[i * n + j]; };
  auto Q = [q, n](int64_t i, int64_t j) -> cplx& { return q[i * n + j]; };
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < n; j++)
      Q(i, j) = i == j ? 1.0 : 0.0;

  // Householder reduction to upper Hessenberg form
  std::vector<cplx> v(n);
  for (int64_t k = 0; k + 2 < n; k++) {
    double norm = 0.0;
    for (int64_t i = k + 1; i < n; i++)
      norm += std::norm(A(i, k));
    norm = std::sqrt(norm);
    if (norm == 0.0)
      continue;
    const cplx x0 = A(k + 1, k);
    const cplx phase = std::abs(x0) > 0 ? x0 / std::abs(x0) : cplx(1.0);
    for (int64_t i = k + 1; i < n; i++)
      v[i] = A(i, k);
    v[k + 1] += phase * norm;
    double vnorm = 0.0;
    for (int64_t i = k + 1; i < n; i++)
      vnorm += std::norm(v[i]);
    if (vnorm == 0.0)
      continue;
    // A <- (I - 2 v v^* / |v|^2) A (I - 2 v v^* / |v|^2), Q <- Q H
    for (int64_t j = 0; j < n; j++) {
      cplx dot = 0.0;
      for (int64_t i = k + 1; i < n; i++)
        dot += std::conj(v[i]) * A(i, j);
      dot *= 2.0 / vnorm;
      for (int64_t i = k + 1; i < n; i++)
        A(i, j) -= v[i] * dot;
    }
    for (int64_t i = 0; i < n; i++) {
      cplx dot = 0.0, dq = 0.0;
      for (int64_t j = k + 1; j < n; j++) {
        dot += A(i, j) * v[j];
        dq += Q(i, j) * v[j];
      }
      dot *= 2.0 / vnorm;
      dq *= 2.0 / vnorm;
      for (int64_t j = k + 1; j < n; j++) {
        A(i, j) -= dot * std::conj(v[j]);
        Q(i, j) -= dq * std::conj(v[j]);
      }
    }
    for (int64_t i = k + 2; i < n; i++)
      A(i, k) = 0.0;
  }

  // Shifted QR on the active window [lo, hi]
  const double eps = std::numeric_limits<double>::epsilon();
  std::vector<double> cs(n);
  std::vector<cplx> sn(n);
  int64_t hi = n - 1;
  int64_t iterations = 0;
  while (hi > 0) {
    int64_t lo = hi;
    while (lo > 0) {
      const double scale = std::abs(A(lo - 1, lo - 1)) + std::abs(A(lo, lo));
      if (std::abs(A(lo, lo - 1)) <= eps * (scale > 0 ? scale : 1.0)) {
        A(lo, lo - 1) = 0.0;
        break;
      }
      lo--;
    }
    if (lo == hi) {
      hi--;
      iterations = 0;
      continue;
    }
    if (++iterations > max_iterations)
      return false;

    // Wilkinson shift: eigenvalue of the trailing 2 x 2 closest to its
    // last diagonal entry (exceptional shifts against cycling)
    cplx mu;
    if (iterations % 11 == 10) {
      mu = A(hi, hi) + 0.75 * std::abs(A(hi, hi - 1));
    } else {
      const cplx p = A(hi - 1, hi - 1), r = A(hi - 1, hi), s = A(hi, hi - 1),
                 d = A(hi, hi);
      const cplx half = 0.5 * (p - d);
      const cplx root = std::sqrt(half * half + r * s);
      const cplx m1 = d - r * s / (half + root), m2 = d - r * s / (half - root);
      const bool ok1 = std::abs(half + root) > 0,
                 ok2 = std::abs(half - root) > 0;
      if (ok1 && (!ok2 || std::abs(m1 - d) <= std::abs(m2 - d)))
        mu = m1;
      else if (ok2)
        mu = m2;
      else
        mu = d;
    }

    for (int64_t i = lo; i <= hi; i++)
      A(i, i) -= mu;
    for (int64_t k = lo; k < hi; k++) {
      givens(A(k, k), A(k + 1, k), cs[k], sn[k]);
      const double c = cs[k];
      const cplx s = sn[k];
      for (int64_t j = k; j < n; j++) {
        const cplx x = A(k, j), y = A(k + 1, j);
        A(k, j) = c * x + s * y;
        A(k + 1, j) = -std::conj(s) * x + c * y;
      }
    }
    for (int64_t k = lo; k < hi; k++) {
      const double c = cs[k];
      const cplx s = sn[k];
      const int64_t rows = std::min(k + 2, hi);
      for (int64_t i = 0; i <= rows; i++) {
        const cplx x = A(i, k), y = A(i, k + 1);
        A(i, k) = c * x + std::conj(s) * y;
        A(i, k + 1) = -s * x + c * y;
      }
      for (int64_t i = 0; i < n; i++) {
        const cplx x = Q(i, k), y = Q(i, k + 1);
        Q(i, k) = c * x + std::conj(s) * y;
        Q(i, k + 1) = -s * x + c * y;
      }
    }
    for (int64_t i = lo; i <= hi; i++)
      A(i, i) += mu;
  }
  for (int64_t i = 1; i < n; i++)
    for (int64_t j = 0; j < i; j++)
      A(i, j) = 0.0;
  return true;
}

} // namespace ttnte::linalg
