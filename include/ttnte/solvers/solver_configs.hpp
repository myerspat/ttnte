#pragma once

#include "ttnte/linalg/format_type.hpp"
#include "ttnte/linalg/tt_config.hpp"
#include "ttnte/solvers/memory_policy.hpp"
#include <memory>

namespace ttnte::solvers {

#ifdef USE_CUDA
inline constexpr bool DEFAULT_USE_GPU = true;
inline constexpr MemoryPolicy DEFAULT_MEMORY_POLICY = MemoryPolicy::RESIDENT;
#else
inline constexpr bool DEFAULT_USE_GPU = false;
inline constexpr MemoryPolicy DEFAULT_MEMORY_POLICY = MemoryPolicy::OUT_OF_CORE;
#endif

enum class ExecMode : uint8_t { SYNC, ASYNC };
enum class CommMode : uint8_t { SYNC, ASYNC };

/// @brief Top-level configuration for the domain decomposition solver. Owned
/// by DDStrategy so that a single strategy object carries all tuning knobs.
///
/// A single DAG iteration covers one complete block-Jacobi sweep. For
/// k-eigenvalue problems convergence requires both the relative flux change
/// and the k change to fall below their respective tolerances.
struct DDSolverConfig {
  /// TT rounding config for boundary communication and intermediate results.
  linalg::TTConfig rounding;
  /// Format for the solution vector
  linalg::FormatType fmt = linalg::FormatType::TENSOR_TRAIN;
  /// Relative flux-change convergence tolerance (per-patch Frobenius norm).
  double tol = 1e-8;
  /// Maximum number of block-Jacobi iterations.
  int max_iter = 100;
  /// Clear assembler memory before solving (frees assembled operators that
  /// are no longer needed once the LinearSystem buffer is built). Fission
  /// operators are retained separately in the driver regardless of this flag.
  bool clear_assemblers = true;
  /// Mode for compute heavy tasks.
  ExecMode exec_mode = ExecMode::ASYNC;
  /// Mode for communication heavy tasks.
  CommMode comm_mode = CommMode::ASYNC;
  /// Number of threads for the TaskScheduler thread pool.
  int num_threads = 4;
  /// Number of CUDA streams for GPU workloads.
  int num_streams = 16;
  /// Use the GPU in compute tasks.
  bool use_gpu = DEFAULT_USE_GPU;
  /// Memory policy for memory management on GPUs.
  MemoryPolicy memory_policy = DEFAULT_MEMORY_POLICY;
  /// Whether to print to terminal.
  bool verbose = false;
  /// Forcing coefficient for the block-Jacobi inner Schwarz tolerance.
  /// The effective inner break tolerance is max(tol, inner_forcing *
  /// outer_error). Larger values exit the inner Schwarz earlier (less work per
  /// outer iteration).
  double inner_forcing = 0.1;
  /// Forcing coefficient for the TT truncation tolerance (AMEn eps and boundary
  /// rounding eps). The effective eps is max(base_eps, eps_forcing *
  /// outer_error). Should be much smaller than inner_forcing so the TT
  /// approximation is accurate even when the Schwarz has not fully converged.
  double eps_forcing = 0.01;

  // =================================================================
  // Constructors
  /// @brief Flat constructor — all parameters supplied directly.
  DDSolverConfig(double tol = 1e-8, int max_iter = 100,
    linalg::FormatType fmt = linalg::FormatType::TENSOR_TRAIN,
    double eps = 1e-12, int64_t max_rank = 500, bool clear_assemblers = true,
    ExecMode exec_mode = ExecMode::ASYNC, CommMode comm_mode = CommMode::ASYNC,
    int num_threads = 4, int num_streams = 16, bool use_gpu = DEFAULT_USE_GPU,
    MemoryPolicy memory_policy = DEFAULT_MEMORY_POLICY,
    double inner_forcing = 0.1, double eps_forcing = 0.01, bool verbose = false)
    : rounding(eps, max_rank), tol(tol), max_iter(max_iter), fmt(fmt),
      clear_assemblers(clear_assemblers), exec_mode(exec_mode),
      comm_mode(comm_mode), num_threads(num_threads), num_streams(num_streams),
      use_gpu(use_gpu), memory_policy(memory_policy),
      inner_forcing(inner_forcing), eps_forcing(eps_forcing), verbose(verbose)
  {}

  /// @brief Rounding-first constructor — pass a pre-built TTConfig.
  DDSolverConfig(linalg::TTConfig rounding, double tol = 1e-8,
    int max_iter = 100,
    linalg::FormatType fmt = linalg::FormatType::TENSOR_TRAIN,
    bool clear_assemblers = true, ExecMode exec_mode = ExecMode::ASYNC,
    CommMode comm_mode = CommMode::ASYNC, int num_threads = 4,
    int num_streams = 16, bool use_gpu = DEFAULT_USE_GPU,
    MemoryPolicy memory_policy = DEFAULT_MEMORY_POLICY,
    double inner_forcing = 0.1, double eps_forcing = 0.01, bool verbose = false)
    : rounding(rounding), tol(tol), max_iter(max_iter),
      clear_assemblers(clear_assemblers), exec_mode(exec_mode),
      comm_mode(comm_mode), num_threads(num_threads), num_streams(num_streams),
      use_gpu(use_gpu), memory_policy(memory_policy),
      inner_forcing(inner_forcing), eps_forcing(eps_forcing), verbose(verbose)
  {}
};

} // namespace ttnte::solvers
