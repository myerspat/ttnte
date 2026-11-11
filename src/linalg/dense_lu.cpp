#include "ttnte/linalg/dense_lu.hpp"
#include "ttnte/linalg/batched_lu.hpp"
#include "ttnte/linalg/lapack.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::linalg::dense {

// LAPACK (dgetrf/dgetrs). A row-major matrix is its transpose in LAPACK's
// column-major order, so dgetrf factors A^T = P L U and the solve of A x = b
// is dgetrs with trans = 'T' on those factors.

int64_t lu_factor(double* a, int64_t n, int64_t* ipiv)
{
  lapack::SingleThreaded single;
  const int m = static_cast<int>(n);
  int* lp = lapack::pivot_buffer(n);
  int info = 0;
  dgetrf_(&m, &m, a, &m, lp, &info);
  if (info < 0)
    throw utils::runtime_error(
      "ttnte::linalg::dense::lu_factor", "dgetrf: bad argument");
  lapack::from_lapack(lp, n, ipiv);
  return info;
}

void lu_solve(const double* a, int64_t n, const int64_t* ipiv, double* b)
{
  const int m = static_cast<int>(n), one = 1;
  int info = 0;
  dgetrs_("T", &m, &one, a, &m, lapack::to_lapack(ipiv, n), b, &m, &info, 1);
}

torch::Tensor batched_lu_factor_(torch::Tensor& a)
{
  if (a.dim() == 3 && a.size(1) != a.size(2)) {
    throw utils::runtime_error(
      "ttnte::linalg::dense::batched_lu_factor_", "Matrices must be square");
  }
  const int64_t n = a.dim() == 3 ? a.size(1) : 0;
  return batched_factor("ttnte::linalg::dense::batched_lu_factor_", a, n,
    [n](double* m, int64_t* piv) { return lu_factor(m, n, piv); });
}

void batched_lu_solve_(
  const torch::Tensor& a, const torch::Tensor& ipiv, torch::Tensor& b)
{
  if (a.dim() == 3 && a.size(1) != a.size(2)) {
    throw utils::runtime_error(
      "ttnte::linalg::dense::batched_lu_solve_", "Matrices must be square");
  }
  const int64_t n = a.dim() == 3 ? a.size(1) : 0;
  batched_solve("ttnte::linalg::dense::batched_lu_solve_", a, ipiv, b,
    [n](const double* m, const int64_t* piv, double* rhs) {
      lu_solve(m, n, piv, rhs);
    });
}

} // namespace ttnte::linalg::dense
