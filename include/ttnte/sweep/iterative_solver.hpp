#pragma once

#include "ttnte/sweep/kronecker_solver.hpp"
#include "ttnte/sweep/sum_factorization.hpp"
#include <atomic>
#include <memory>
#include <vector>

namespace ttnte::sweep {

/// @brief Local solver of curved patches without a factorization: GMRES on
/// the matrix-free, sum-factorized operator, right-preconditioned by the
/// Kronecker-sum solver of the patch's parametric twin (the same B-spline
/// bases with the metric averaged over the patch; see
/// KroneckerSolver::parametric_twin()), in the spirit of the fast
/// diagonalization preconditioners of Sangalli and Tani (SISC 38, 2016).
class IterativeSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<IterativeSolver>;

  /// @brief Scratch of one worker.
  struct Workspace {
    SumFactorization::Workspace apply;
    KroneckerSolver::Workspace twin;
    std::vector<double> t;
  };

private:
  // =================================================================
  // Private data
  SumFactorization::Ptr op_;
  KroneckerSolver::Ptr twin_;
  /// (ndim, ndim) patch average of |det J| J^{-1}, and of |det J|.
  std::vector<double> metric_;
  double measure_ = 1.0;
  int64_t ndim_ = 0;
  double tol_ = 1e-10;
  int64_t restart_ = 20;
  int64_t max_iter_ = 200;
  /// Statistics (all threads).
  mutable std::atomic<int64_t> solves_ {0};
  mutable std::atomic<int64_t> iterations_ {0};
  mutable std::atomic<int64_t> failures_ {0};

  // =================================================================
  // Protected constructors
  IterativeSolver(const PatchDiscretization& disc, double tol, int64_t restart,
    int64_t max_iter);

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new IterativeSolver(std::forward<Args>(args)...));
  }

  /// @brief Solve L(omega, sigma) x = b in place.
  /// @return Whether GMRES converged (b is unspecified otherwise; the caller
  /// falls back to a direct solve).
  bool solve(
    const double* omega, double sigma, double* b, Workspace& work) const;

  /// @brief Apply the patch's operator matrix-free: y = L(omega, sigma) u.
  void apply(const double* omega, double sigma, const double* u, double* y,
    Workspace& work) const
  {
    op_->apply(omega, sigma, u, y, work.apply);
  }

  // =================================================================
  // Public getters
  int64_t num_solves() const noexcept { return solves_; }
  int64_t num_iterations() const noexcept { return iterations_; }
  int64_t num_failures() const noexcept { return failures_; }
};

} // namespace ttnte::sweep
