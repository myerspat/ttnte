#include "ttnte/solvers/amen_solver.hpp"
#include "ttnte/linalg/tt_operator.hpp"
#include "ttnte/linalg/tt_ops.hpp"
#include "ttnte/linalg/tt_state.hpp"

namespace ttnte::solvers {

// =================================================================
// Public methods
void AMEnSolver::solve(const linalg::LinearSystem::Ptr& local_system)
{
  // Cast the linear system pointer to a TT linear system
  assert(std::dynamic_pointer_cast<linalg::TTLinearSystem>(local_system));

  linalg::TTOperator::Ptr A = std::static_pointer_cast<linalg::TTOperator>(
    local_system->get_interior_op());
  linalg::TTState::Ptr x0 =
    std::static_pointer_cast<linalg::TTState>(local_system->get_state());
  linalg::TTState::Ptr b =
    std::static_pointer_cast<linalg::TTState>(local_system->get_source());

  // Run AMEn solver
  const auto x = linalg::amen_solve(A, b, x0, nswp_, eps_, max_rank_, max_full_,
    kickrank_, kick2_, local_iterations_, resets_, verbose_, preconditioner_);

  // Update the linear system
  local_system->set_state(x);
}

} // namespace ttnte::solvers
