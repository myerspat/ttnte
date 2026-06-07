#include "ttnte/solvers/amen_solver.hpp"
#include "ttnte/linalg/ops.hpp"

namespace ttnte::solvers {

// =================================================================
// Public methods
void AMEnSolver::solve(const linalg::LinearSystem::Ptr& local_system)
{
  // Get the operators for the linear system
  linalg::Operator A = local_system->get_interior_op();
  linalg::State x0 = local_system->get_state();
  linalg::State b = local_system->get_source();

  // Run AMEn solver
  const auto x = linalg::amen_solve(A, b, x0, nswp_, eps_, max_rank_, max_full_,
    kickrank_, kick2_, local_iterations_, resets_, verbose_, preconditioner_);

  // Update the linear system
  local_system->set_state(x);
}

} // namespace ttnte::solvers
