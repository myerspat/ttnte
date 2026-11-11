#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

// =================================================================
// LAPACK / BLAS routines used by ttnte::linalg (Fortran interface, LP64).
//
// The extension links the system LAPACK found by CMake (find_package(LAPACK);
// OpenBLAS on the development machine, liblapack-dev in CI). torch's bundled
// LAPACK exports only some routines (no band or Schur routines), so these are
// declared here directly. Character arguments carry the hidden length
// arguments of the gfortran calling convention.

extern "C" {
void dgbtrf_(const int* m, const int* n, const int* kl, const int* ku,
  double* ab, const int* ldab, int* ipiv, int* info);
void dgbtrs_(const char* trans, const int* n, const int* kl, const int* ku,
  const int* nrhs, const double* ab, const int* ldab, const int* ipiv,
  double* b, const int* ldb, int* info, std::size_t trans_len);
void zgbtrf_(const int* m, const int* n, const int* kl, const int* ku,
  std::complex<double>* ab, const int* ldab, int* ipiv, int* info);
void zgbtrs_(const char* trans, const int* n, const int* kl, const int* ku,
  const int* nrhs, const std::complex<double>* ab, const int* ldab,
  const int* ipiv, std::complex<double>* b, const int* ldb, int* info,
  std::size_t trans_len);
void dgetrf_(
  const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a,
  const int* lda, const int* ipiv, double* b, const int* ldb, int* info,
  std::size_t trans_len);
void dpbtrf_(const char* uplo, const int* n, const int* kd, double* ab,
  const int* ldab, int* info, std::size_t uplo_len);
void dtbmv_(const char* uplo, const char* trans, const char* diag, const int* n,
  const int* k, const double* a, const int* lda, double* x, const int* incx,
  std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len);
void dtbsv_(const char* uplo, const char* trans, const char* diag, const int* n,
  const int* k, const double* a, const int* lda, double* x, const int* incx,
  std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len);
void zgees_(const char* jobvs, const char* sort,
  int (*select)(const std::complex<double>*), const int* n,
  std::complex<double>* a, const int* lda, int* sdim, std::complex<double>* w,
  std::complex<double>* vs, const int* ldvs, std::complex<double>* work,
  const int* lwork, double* rwork, int* bwork, int* info, std::size_t jobvs_len,
  std::size_t sort_len);
/// OpenBLAS only (weak: absent with other LAPACK providers).
void openblas_set_num_threads(int num_threads) __attribute__((weak));
/// oneMKL only (weak): the calling thread's MKL thread count (0: the global
/// one); returns the previous value. The C interface: the lowercase
/// mkl_set_num_threads_local symbol is MKL's Fortran one (takes int*).
int MKL_Set_Num_Threads_Local(int num_threads) __attribute__((weak));
}

namespace ttnte::linalg::lapack {

/// @brief Keep LAPACK and the BLAS it calls single-threaded for the
/// lifetime of the guard: these routines run inside ttnte's own parallel
/// tasks (one small system per task), where a threaded BLAS would
/// oversubscribe the cores.
///
/// Two providers are involved in one process: the band and Schur routines
/// come from the linked LAPACK (OpenBLAS here), but their BLAS-3 calls
/// (dgemm_, dtrsm_, ...) and dgetrf_/dgetrs_ bind to torch's copies, loaded
/// first (oneMKL with GNU OpenMP threading). Without this guard a blocked
/// dgbtrf inside a sweep task started an MKL thread team per task (2.5x
/// slower on 8 threads, 48 -> 104 OS threads) unless OMP_NUM_THREADS=1.
/// MKL's setting is per calling thread and restored on exit, so other
/// threads and OpenMP (at::parallel_for) are untouched; OpenBLAS is set to
/// one thread once. Both are no-ops for other providers. Used around the
/// factorizations and the Schur form; the triangular band solves call only
/// BLAS-1/2 and skip it (its two MKL calls cost 7-15% of a cached sweep).
class SingleThreaded {
public:
  SingleThreaded()
  {
    static std::once_flag once;
    std::call_once(once, [] {
      if (openblas_set_num_threads)
        openblas_set_num_threads(1);
    });
    if (MKL_Set_Num_Threads_Local)
      previous_ = MKL_Set_Num_Threads_Local(1);
  }
  ~SingleThreaded()
  {
    if (MKL_Set_Num_Threads_Local)
      MKL_Set_Num_Threads_Local(previous_);
  }
  SingleThreaded(const SingleThreaded&) = delete;
  SingleThreaded& operator=(const SingleThreaded&) = delete;

private:
  int previous_ = 0;
};

/// @brief LAPACK's 1-based int pivots from ttnte's 0-based int64 ones, in a
/// per-thread buffer.
inline const int* to_lapack(const int64_t* ipiv, int64_t n)
{
  thread_local std::vector<int> buffer;
  buffer.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; i++)
    buffer[i] = static_cast<int>(ipiv[i] + 1);
  return buffer.data();
}

/// @brief Per-thread buffer for LAPACK pivots; from_lapack() converts back.
inline int* pivot_buffer(int64_t n)
{
  thread_local std::vector<int> buffer;
  buffer.resize(static_cast<size_t>(n));
  return buffer.data();
}

inline void from_lapack(const int* lp, int64_t n, int64_t* ipiv)
{
  for (int64_t i = 0; i < n; i++)
    ipiv[i] = lp[i] - 1;
}

} // namespace ttnte::linalg::lapack
