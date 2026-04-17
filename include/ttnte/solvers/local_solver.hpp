#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include <memory>

namespace ttnte::solvers {

class LocalSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<LocalSolver>;

  virtual ~LocalSolver() = default;

  // =================================================================
  // Public methods
  virtual void solve(const linalg::LinearSystem::Ptr& local_system) = 0;
};

} // namespace ttnte::solvers
