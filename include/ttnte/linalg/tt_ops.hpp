#pragma once

#include "ttnte/linalg/tt_engine.hpp"
#include <limits>
#include <optional>

namespace ttnte::linalg {

/// @brief Compute a matrix-matrix product in TT format.
/// @brief a The TT-matrix.
/// @brief b The TT-matrix.
/// @return The exact solution to a @ b.
TTEngine mm(const TTEngine& a, const TTEngine& b);

/// @brief Compute a matrix-vector product in TT format.
/// @brief a The TT-matrix.
/// @brief b The TT-vector.
/// @return The exact solution to a @ b.
inline TTEngine mv(const TTEngine& a, const TTEngine& b)
{
  return mm(a, b);
}

/// @brief Perform an element-wise division with two tensor trains using AMEn.
/// This calls the torchTT implementation `torchtt._division.amen_divide()` in
/// Python.
/// @param a The numerator TT.
/// @param b The denominator TT.
/// @param nswp The maximum number of sweeps.
/// @param initial_guess The initial guess of the solution.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @param max_full The maximum allowed size of a local problem to use a direct
/// solver. For anything greater we use GMRES.
/// @param kickrank The enrichment rank size.
/// @param kick2 ALS enrichment.
/// @param trunc_norm Which norm to base convergence off.
/// @param local_iterations The max number of GMRES iterations.
/// @param resets The maximum number of restarts in GMRES.
/// @param verbose Whether to print progress.
/// @param preconditioner Which preconditioner to use.
/// @return The approximate solution for a / b.
TTEngine elementwise_divide(const TTEngine& a, const TTEngine& b, int nswp = 50,
  std::optional<TTEngine> initial_guess = std::nullopt, double eps = 1e-12,
  int max_rank = std::numeric_limits<int>::max(), int max_full = 500,
  int kickrank = 4, int kick2 = 0, std::string trunc_norm = "res",
  int local_iterations = 40, int resets = 2, bool verbose = false,
  std::optional<std::string> preconditioner = std::nullopt);

/// @brief Compute a matrix-vector product in TT format using DMRG. This calls
/// the C++ torchTT implementation.
/// @param A The TT-matrix.
/// @param x The TT-vector.
/// @param y0 The initial guess TT-vector.
/// @param nswp The maximum number of sweeps.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @param kickrank The enrichment rank size.
/// @param verbose Whether to print progress.
/// @return The approximate solution to A @ x.
TTEngine dmrg_mv(const TTEngine& A, const TTEngine& x,
  std::optional<TTEngine> y0 = std::nullopt, int nswp = 20, double eps = 1e-12,
  int max_rank = std::numeric_limits<int>::max(), int kickrank = 4,
  bool verbose = false);

/// @brief Compute the element-wise (Hadamard) product using that defined in
/// https://arxiv.org/pdf/2410.19747. This is functionally equivalent to
/// `(a * b).round(eps, max_rank)`.
/// @param a The TT-matrix.
/// @param b The TT-vector.
/// @param y0 The initial guess TT-vector.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @return The approximate solution to a * b.
TTEngine fast_hadamard(const TTEngine& a, const TTEngine& b, double eps = 1e-10,
  int max_rank = std::numeric_limits<int>::max());

/// @brief Compute the matrix-matrix product in TT format using that defined in
/// https://arxiv.org/pdf/2410.19747. This is functionally equivalent to
/// `(a @ b).round(eps, max_rank)`.
/// @param a The TT-matrix.
/// @param b The TT-matrix.
/// @param y0 The initial guess TT-vector.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @return The approximate solution to a @ b.
TTEngine fast_mm(const TTEngine& a, const TTEngine& b, double eps = 1e-10,
  int max_rank = std::numeric_limits<int>::max());

/// @brief Compute the matrix-vector product in TT format using that defined in
/// https://arxiv.org/pdf/2410.19747. This is functionally equivalent to
/// `(a @ b).round(eps, max_rank)`.
/// @param a The TT-matrix.
/// @param b The TT-vector.
/// @param y0 The initial guess TT-vector.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @return The approximate solution to a @ b.
inline TTEngine fast_mv(const TTEngine& a, const TTEngine& b,
  double eps = 1e-10, int max_rank = std::numeric_limits<int>::max())
{
  return fast_mm(a, b, eps, max_rank);
}

/// @brief Compute the matrix-matrix product in TT format using AMEn. This calls
/// `torchtt._amen.amen_mm()`.
/// @param a The TT-matrix.
/// @param b The TT-matrix.
/// @param nswp The maximum number of sweeps.
/// @param x0 The initial guess TT-matrix.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @param kickrank The enrichment rank size.
/// @param kick2 ALS enrichment.
/// @param verbose Whether to print progress.
/// @return The approximate solution to a @ b.
TTEngine amen_mm(const linalg::TTEngine& a, const linalg::TTEngine& b,
  int nswp = 22, std::optional<linalg::TTEngine> x0 = std::nullopt,
  double eps = 1e-10, int max_rank = std::numeric_limits<int>::max(),
  int kickrank = 4, int kick2 = 0, bool verbose = false);

/// @brief Compute the matrix-vector product in TT format using AMEn. This calls
/// `torchtt._amen.amen_mm()`.
/// @param a The TT-matrix.
/// @param b The TT-vector.
/// @param nswp The maximum number of sweeps.
/// @param x0 The initial guess TT-vector.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @param kickrank The enrichment rank size.
/// @param kick2 ALS enrichment.
/// @param verbose Whether to print progress.
/// @return The approximate solution to a @ b.
inline TTEngine amen_mv(const linalg::TTEngine& a, const linalg::TTEngine& b,
  int nswp = 22, std::optional<linalg::TTEngine> x0 = std::nullopt,
  double eps = 1e-10, int max_rank = std::numeric_limits<int>::max(),
  int kickrank = 4, int kick2 = 0, bool verbose = false)
{
  return amen_mm(a, b, nswp, x0, eps, max_rank, kickrank, kick2, verbose);
}

/// @brief Solve a linear system in TT format using the Alternating Minimal
/// Energy Method (AMEn). This calls the C++ implementation provided by torchTT.
/// @param A The TT-matrix.
/// @param b The TT-vector.
/// @param x0 The initial guess TT-vector.
/// @param nswp The maximum number of sweeps.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @param max_full The maximum allowed size of a local problem to use a direct
/// solver. For anything greater we use GMRES.
/// @param kickrank The enrichment rank size.
/// @param kick2 ALS enrichment.
/// @param local_iterations The max number of GMRES iterations.
/// @param resets The maximum number of restarts in GMRES.
/// @param verbose Whether to print progress.
/// @param preconditioner Which preconditioner to use.
/// @return The approximate solution x for A @ x = b.
TTEngine amen_solve(const linalg::TTEngine& A, const linalg::TTEngine& b,
  std::optional<linalg::TTEngine> x0 = std::nullopt, int nswp = 22,
  double eps = 1e-10, int max_rank = std::numeric_limits<int>::max(),
  int max_full = 500, int kickrank = 4, int kick2 = 0,
  int local_iterations = 40, int resets = 2, bool verbose = false,
  int preconditioner = 0);

/// @brief Interpolate a TT-vector representation for a function using TT-cross
/// (univariate case). This calls the Python implementation in
/// `torchtt.interpolate.function_interpolate()`.
/// @param func The function to interpolate into the TT format.
/// @param x The points to evaluate the function at.
/// @param eps The truncation tolerance.
/// @param start_tens The initial guess TT-vector.
/// @param nswp The maximum number of sweeps.
/// @param kick The enrichment rank size.
/// @param rmax The maximum allowed rank.
/// @param verbose Whether to print progress.
/// @return The interpolated TT-vector.
TTEngine function_interpolate(
  const std::function<torch::Tensor(const torch::Tensor&)>& func,
  const TTEngine& x, double eps = 1e-9,
  std::optional<TTEngine> start_tens = std::nullopt, int nswp = 20,
  int kick = 2, int rmax = std::numeric_limits<int>::max(),
  bool verbose = false);

/// @brief Interpolate a TT-vector representation for a function using TT-cross
/// (multivariate case). This calls the Python implementation in
/// `torchtt.interpolate.function_interpolate()`.
/// @param func The function to interpolate into the TT format.
/// @param x The points to evaluate the function at.
/// @param eps The truncation tolerance.
/// @param start_tens The initial guess TT-vector.
/// @param nswp The maximum number of sweeps.
/// @param kick The enrichment rank size.
/// @param rmax The maximum allowed rank.
/// @param verbose Whether to print progress.
/// @return The interpolated TT-vector.
TTEngine function_interpolate(
  const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>& func,
  const std::vector<TTEngine>& xs, double eps = 1e-9,
  std::optional<TTEngine> start_tens = std::nullopt, int nswp = 20,
  int kick = 2, int rmax = std::numeric_limits<int>::max(),
  bool verbose = false);

/// @brief Interpolate a TT-vector representation of a tensor or function with
/// index inputs using DMRG. This calls the Python implementation in
/// `torchtt.interpolate.dmrg_cross()`.
/// @param func The function to interpolate into the TT format.
/// @param N The size of each dimension.
/// @param eps The truncation tolerance.
/// @param nswp The maximum number of sweeps.
/// @param x0 The initial guess TT-vector.
/// @param kick The enrichment rank size.
/// @param rmax The maximum allowed rank.
/// @param verbose Whether to print progress.
/// @param device Which device to run this on.
/// @param dtype What data type to compute this.
/// @return The interpolated TT-vector.
TTEngine dmrg_cross(
  const std::function<torch::Tensor(const torch::Tensor&)>& func,
  const std::vector<int64_t>& N, double eps = 1e-9, int nswp = 20,
  std::optional<TTEngine> x0 = std::nullopt, int kick = 2,
  int rmax = std::numeric_limits<int>::max(), bool verbose = false,
  const torch::Device& device = torch::kCPU,
  const torch::ScalarType& dtype = torch::kFloat64);

} // namespace ttnte::linalg
