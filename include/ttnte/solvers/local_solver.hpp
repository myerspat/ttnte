#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include <memory>

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
};

} // namespace ttnte::solvers
