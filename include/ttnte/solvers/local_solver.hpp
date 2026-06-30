#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include <memory>
#include <tuple>

namespace ttnte::solvers {

/// @brief The local solver base class.
class LocalSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<LocalSolver>;

  virtual ~LocalSolver() = default;

  // =================================================================
  // Public methods
  /// @brief Solve the local linear system.
  /// @param local_system The local linear system to be solved.
  virtual void solve(const linalg::LinearSystem::Ptr& local_system) = 0;

  /// @brief Compute the right-hand-side of the linear system based on the new
  /// boundary conditions.
  /// @param local_system The local linear system to be solved.
  /// @return A tuple of A, b, x0.
  std::tuple<linalg::Operator, linalg::State, linalg::State> presolve(
    const linalg::LinearSystem::Ptr& sys) const;

  /// @brief Post solve cleanup.
  /// @param local_system The local linear system to be solved.
  void postsolve(
    const linalg::LinearSystem::Ptr& sys, const linalg::State& x) const;

  // =================================================================
  // Public getters / setters
  /// @return The current truncation tolerance of the solver.
  virtual double get_eps() const noexcept { return 0.0; }
  /// @param eps The new truncation tolerance of the solver.
  virtual void set_eps(double eps) {}
  /// @return The maximum rank.
  virtual int get_max_rank() const noexcept
  {
    return std::numeric_limits<int>::max();
  }
  /// @param max_rank The new maximum rank of the solver.
  virtual void set_max_rank(int max_rank) {}
};

} // namespace ttnte::solvers
