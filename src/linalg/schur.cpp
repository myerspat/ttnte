#include "ttnte/linalg/schur.hpp"
#include "ttnte/linalg/lapack.hpp"
#include <algorithm>
#include <vector>

namespace ttnte::linalg {

bool complex_schur(int64_t n, std::complex<double>* a, std::complex<double>* q,
  int64_t max_iterations)
{
  (void)max_iterations; // LAPACK's own iteration limit applies
  using cplx = std::complex<double>;
  lapack::SingleThreaded single;
  if (n == 0)
    return true;
  // Row-major in and out; zgees works on the column-major transpose
  const int m = static_cast<int>(n);
  std::vector<cplx> A(n * n), vs(n * n), w(n);
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < n; j++)
      A[i + j * n] = a[i * n + j];
  std::vector<double> rwork(n);
  int sdim = 0, info = 0, lwork = -1;
  cplx query;
  zgees_("V", "N", nullptr, &m, A.data(), &m, &sdim, w.data(), vs.data(), &m,
    &query, &lwork, rwork.data(), nullptr, &info, 1, 1);
  lwork = std::max(1, static_cast<int>(query.real()));
  std::vector<cplx> work(lwork);
  zgees_("V", "N", nullptr, &m, A.data(), &m, &sdim, w.data(), vs.data(), &m,
    work.data(), &lwork, rwork.data(), nullptr, &info, 1, 1);
  if (info != 0)
    return false;
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < n; j++) {
      a[i * n + j] = A[i + j * n];
      q[i * n + j] = vs[i + j * n];
    }
  return true;
}

} // namespace ttnte::linalg
