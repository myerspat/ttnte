#pragma once

#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/state.hpp"
#include "ttnte/linalg/tt_ops.hpp"
#include <variant>

/// @file
/// @brief High-level tensor train operations mapping `Operator` and `State`
/// objects to their underlying engine implementations.

namespace ttnte::linalg {

/// @brief Compute a matrix-matrix product in TT format.
/// @param a The TT-matrix (Operator).
/// @param b The TT-matrix (Operator).
/// @return The exact solution to a @ b.
inline Operator mm(const Operator& a, const Operator& b)
{
  return std::visit(
    [](const auto& v0, const auto& v1) -> Operator {
      using TypeA = std::decay_t<decltype(v0)>;
      using TypeB = std::decay_t<decltype(v1)>;

      if constexpr (std::is_same_v<TypeA, TypeB>) {
        // Because of C++ overload resolution, this automatically routes to:
        // - ttnte::linalg::mm(TTEngine, TTEngine) OR
        // - ttnte::linalg::mm(DenseEngine, DenseEngine)
        return Operator(mm(v0, v1));

      } else {
        throw utils::runtime_error("ttnte::linalg::mm",
          "Matrix multiplication between different "
          "backend formats is not supported.");
      }
    },
    a.get_variant(), b.get_variant());
}

/// @brief Compute a matrix-vector product in TT format.
/// @param a The TT-matrix (Operator).
/// @param b The TT-vector (State).
/// @return The exact solution to a @ b.
inline State mv(const Operator& a, const State& b)
{
  return std::visit(
    [](const auto& v0, const auto& v1) -> State {
      using TypeA = std::decay_t<decltype(v0)>;
      using TypeB = std::decay_t<decltype(v1)>;

      if constexpr (std::is_same_v<TypeA, TypeB>) {
        return State(mv(v0, v1));

      } else {
        throw utils::runtime_error(
          "ttnte::linalg::mm", // intentionally kept as "mm" per original, or
                               // update to "mv"
          "Matrix multiplication between different "
          "backend formats is not supported.");
      }
    },
    a.get_variant(), b.get_variant());
}

/// @brief Perform an element-wise division with two tensor trains using AMEn.
/// This calls the torchTT implementation `torchtt._division.amen_divide()` in
/// Python.
/// @param a The numerator TT-vector (State).
/// @param b The denominator TT-vector (State).
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
inline State elementwise_divide(const State& a, const State& b, int nswp = 50,
  std::optional<State> initial_guess = std::nullopt, double eps = 1e-12,
  int max_rank = std::numeric_limits<int>::max(), int max_full = 500,
  int kickrank = 4, int kick2 = 0, std::string trunc_norm = "res",
  int local_iterations = 40, int resets = 2, bool verbose = false,
  std::optional<std::string> preconditioner = std::nullopt)
{
  std::optional<TTEngine> guess_engine = std::nullopt;
  if (initial_guess.has_value()) {
    guess_engine = initial_guess->as_tt();
  }

  TTEngine result = elementwise_divide(a.as_tt(), b.as_tt(), nswp, guess_engine,
    eps, max_rank, max_full, kickrank, kick2, std::move(trunc_norm),
    local_iterations, resets, verbose, std::move(preconditioner));

  return State(std::move(result));
}

/// @brief Compute a matrix-vector product in TT format using DMRG. This calls
/// the C++ torchTT implementation.
/// @param A The TT-matrix (Operator).
/// @param x The TT-vector (State).
/// @param y0 The initial guess TT-vector (State).
/// @param nswp The maximum number of sweeps.
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @param kickrank The enrichment rank size.
/// @param verbose Whether to print progress.
/// @return The approximate solution to A @ x.
inline State dmrg_mv(const Operator& A, const State& x,
  std::optional<State> y0 = std::nullopt, int nswp = 20, double eps = 1e-12,
  int max_rank = std::numeric_limits<int>::max(), int kickrank = 4,
  bool verbose = false)
{
  std::optional<TTEngine> y0_engine = std::nullopt;
  if (y0.has_value()) {
    y0_engine = y0->as_tt();
  }

  TTEngine result = dmrg_mv(
    A.as_tt(), x.as_tt(), y0_engine, nswp, eps, max_rank, kickrank, verbose);

  return State(std::move(result));
}

/// @brief Compute the element-wise (Hadamard) product using that defined in
/// https://arxiv.org/pdf/2410.19747. This is functionally equivalent to `(a *
/// b).round(eps, max_rank)`.
/// @param a The TT-matrix (Operator).
/// @param b The TT-vector (State).
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @return The approximate solution to a * b.
inline State fast_hadamard(const Operator& a, const State& b,
  double eps = 1e-10, int max_rank = std::numeric_limits<int>::max())
{
  TTEngine result = fast_hadamard(a.as_tt(), b.as_tt(), eps, max_rank);
  return State(std::move(result));
}

/// @brief Compute the matrix-matrix product in TT format using that defined in
/// https://arxiv.org/pdf/2410.19747. This is functionally equivalent to `(a @
/// b).round(eps, max_rank)`.
/// @param a The TT-matrix (Operator).
/// @param b The TT-matrix (Operator).
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @return The approximate solution to a @ b.
inline Operator fast_mm(const Operator& a, const Operator& b,
  double eps = 1e-10, int max_rank = std::numeric_limits<int>::max())
{
  TTEngine result = fast_mm(a.as_tt(), b.as_tt(), eps, max_rank);
  return Operator(std::move(result));
}

/// @brief Compute the matrix-vector product in TT format using that defined in
/// https://arxiv.org/pdf/2410.19747. This is functionally equivalent to `(a @
/// b).round(eps, max_rank)`.
/// @param a The TT-matrix (Operator).
/// @param b The TT-vector (State).
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @return The approximate solution to a @ b.
inline State fast_mv(const Operator& a, const State& b, double eps = 1e-10,
  int max_rank = std::numeric_limits<int>::max())
{
  TTEngine result = fast_mv(a.as_tt(), b.as_tt(), eps, max_rank);
  return State(std::move(result));
}

/// @brief Compute the matrix-matrix product in TT format using AMEn. This calls
/// `torchtt._amen.amen_mm()`.
/// @param a The TT-matrix (Operator).
/// @param b The TT-matrix (Operator).
/// @param nswp The maximum number of sweeps.
/// @param x0 The initial guess TT-matrix (Operator).
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @param kickrank The enrichment rank size.
/// @param kick2 ALS enrichment.
/// @param verbose Whether to print progress.
/// @return The approximate solution to a @ b.
inline Operator amen_mm(const Operator& a, const Operator& b, int nswp = 22,
  std::optional<Operator> x0 = std::nullopt, double eps = 1e-10,
  int max_rank = std::numeric_limits<int>::max(), int kickrank = 4,
  int kick2 = 0, bool verbose = false)
{
  std::optional<TTEngine> x0_engine = std::nullopt;
  if (x0.has_value()) {
    x0_engine = x0->as_tt();
  }

  TTEngine result = amen_mm(a.as_tt(), b.as_tt(), nswp, x0_engine, eps,
    max_rank, kickrank, kick2, verbose);

  return Operator(std::move(result));
}

/// @brief Compute the matrix-vector product in TT format using AMEn. This calls
/// `torchtt._amen.amen_mm()`.
/// @param a The TT-matrix (Operator).
/// @param b The TT-vector (State).
/// @param nswp The maximum number of sweeps.
/// @param x0 The initial guess TT-vector (State).
/// @param eps The truncation tolerance.
/// @param max_rank The maximum allowed rank.
/// @param kickrank The enrichment rank size.
/// @param kick2 ALS enrichment.
/// @param verbose Whether to print progress.
/// @return The approximate solution to a @ b.
inline State amen_mv(const Operator& a, const State& b, int nswp = 22,
  std::optional<State> x0 = std::nullopt, double eps = 1e-10,
  int max_rank = std::numeric_limits<int>::max(), int kickrank = 4,
  int kick2 = 0, bool verbose = false)
{
  std::optional<TTEngine> x0_engine = std::nullopt;
  if (x0.has_value()) {
    x0_engine = x0->as_tt();
  }

  TTEngine result = amen_mv(a.as_tt(), b.as_tt(), nswp, x0_engine, eps,
    max_rank, kickrank, kick2, verbose);

  return State(std::move(result));
}

/// @brief Solve a linear system in TT format using the Alternating Minimal
/// Energy Method (AMEn). This calls the C++ implementation provided by torchTT.
/// @param A The TT-matrix (Operator).
/// @param b The TT-vector (State).
/// @param x0 The initial guess TT-vector (State).
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
inline State amen_solve(const Operator& A, const State& b,
  std::optional<State> x0 = std::nullopt, int nswp = 22, double eps = 1e-10,
  int max_rank = std::numeric_limits<int>::max(), int max_full = 500,
  int kickrank = 4, int kick2 = 0, int local_iterations = 40, int resets = 2,
  bool verbose = false, int preconditioner = 0)
{
  std::optional<TTEngine> x0_engine = std::nullopt;
  if (x0.has_value()) {
    x0_engine = x0->as_tt();
  }

  TTEngine result =
    amen_solve(A.as_tt(), b.as_tt(), x0_engine, nswp, eps, max_rank, max_full,
      kickrank, kick2, local_iterations, resets, verbose, preconditioner);

  return State(std::move(result));
}

/// @brief Interpolate a TT-vector representation for a function using TT-cross
/// (univariate case). This calls the Python implementation in
/// `torchtt.interpolate.function_interpolate()`.
/// @param func The function to interpolate into the TT format.
/// @param x The points to evaluate the function at (State).
/// @param eps The truncation tolerance.
/// @param start_tens The initial guess TT-vector (State).
/// @param nswp The maximum number of sweeps.
/// @param kick The enrichment rank size.
/// @param rmax The maximum allowed rank.
/// @param verbose Whether to print progress.
/// @return The interpolated TT-vector as a State.
inline State function_interpolate(
  const std::function<torch::Tensor(const torch::Tensor&)>& func,
  const State& x, double eps = 1e-9,
  std::optional<State> start_tens = std::nullopt, int nswp = 20, int kick = 2,
  int rmax = std::numeric_limits<int>::max(), bool verbose = false)
{
  std::optional<TTEngine> start_engine = std::nullopt;
  if (start_tens.has_value()) {
    start_engine = start_tens->as_tt();
  }

  TTEngine result = function_interpolate(
    func, x.as_tt(), eps, start_engine, nswp, kick, rmax, verbose);

  return State(std::move(result));
}

/// @brief Interpolate a TT-vector representation for a function using TT-cross
/// (multivariate case). This calls the Python implementation in
/// `torchtt.interpolate.function_interpolate()`.
/// @param func The function to interpolate into the TT format.
/// @param xs The points to evaluate the function at (vector of States).
/// @param eps The truncation tolerance.
/// @param start_tens The initial guess TT-vector (State).
/// @param nswp The maximum number of sweeps.
/// @param kick The enrichment rank size.
/// @param rmax The maximum allowed rank.
/// @param verbose Whether to print progress.
/// @return The interpolated TT-vector as a State.
inline State function_interpolate(
  const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>& func,
  const std::vector<State>& xs, double eps = 1e-9,
  std::optional<State> start_tens = std::nullopt, int nswp = 20, int kick = 2,
  int rmax = std::numeric_limits<int>::max(), bool verbose = false)
{
  // Manually map vector of high-level States to vector of low-level TTEngines
  std::vector<TTEngine> xs_engines;
  xs_engines.reserve(xs.size());
  for (const auto& state : xs) {
    xs_engines.push_back(state.as_tt());
  }

  std::optional<TTEngine> start_engine = std::nullopt;
  if (start_tens.has_value()) {
    start_engine = start_tens->as_tt();
  }

  TTEngine result = function_interpolate(
    func, xs_engines, eps, start_engine, nswp, kick, rmax, verbose);

  return State(std::move(result));
}

/// @brief Interpolate a TT-vector representation of a tensor or function with
/// index inputs using DMRG. This calls the Python implementation in
/// `torchtt.interpolate.dmrg_cross()`.
/// @param func The function to interpolate into the TT format.
/// @param N The size of each dimension.
/// @param eps The truncation tolerance.
/// @param nswp The maximum number of sweeps.
/// @param x0 The initial guess TT-vector (State).
/// @param kick The enrichment rank size.
/// @param rmax The maximum allowed rank.
/// @param verbose Whether to print progress.
/// @param device Which device to run this on.
/// @param dtype What data type to compute this.
/// @return The interpolated TT-vector as a State.
inline State dmrg_cross(
  const std::function<torch::Tensor(const torch::Tensor&)>& func,
  const std::vector<int64_t>& N, double eps = 1e-9, int nswp = 20,
  std::optional<State> x0 = std::nullopt, int kick = 2,
  int rmax = std::numeric_limits<int>::max(), bool verbose = false,
  const torch::Device& device = torch::kCPU,
  const torch::ScalarType& dtype = torch::kFloat64)
{
  std::optional<TTEngine> x0_engine = std::nullopt;
  if (x0.has_value()) {
    x0_engine = x0->as_tt();
  }

  TTEngine result = dmrg_cross(
    func, N, eps, nswp, x0_engine, kick, rmax, verbose, device, dtype);

  return State(std::move(result));
}

} // namespace ttnte::linalg
