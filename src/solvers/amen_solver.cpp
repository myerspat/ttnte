#include "ttnte/solvers/amen_solver.hpp"
#include "ttnte/linalg/tt_operator.hpp"
#include "ttnte/linalg/tt_state.hpp"

#include "amen_solve.h"

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

  // Get cores for each portion
  std::vector<torch::Tensor> A_cores(
    A->get_cores().begin(), A->get_cores().end());
  std::vector<torch::Tensor> x0_cores(
    x0->get_cores().begin(), x0->get_cores().end());
  std::vector<torch::Tensor> b_cores(
    b->get_cores().begin(), b->get_cores().end());

  // Vector of sizes
  std::vector<uint64_t> N;
  std::vector<uint64_t> rA;
  std::vector<uint64_t> rb;
  std::vector<uint64_t> rx0;

  int64_t ndim = A_cores.size();
  N.reserve(ndim);
  rA.reserve(ndim + 1);
  rb.reserve(ndim + 1);
  rx0.reserve(ndim + 1);

  // Squeeze the cores for the vector
  for (size_t i = 0; i < ndim; i++) {
    N.push_back(A_cores[i].size(2));
    rA.push_back(A_cores[i].size(0));
    rb.push_back(b_cores[i].size(0));
    rx0.push_back(x0_cores[i].size(0));

    x0_cores[i] = x0_cores[i].squeeze(2);
    b_cores[i] = b_cores[i].squeeze(2);
  }
  rA.push_back(1);
  rb.push_back(1);
  rx0.push_back(1);

  // Run torchTT AMEn solver
  auto x_cores = amen_solve(A_cores, b_cores, x0_cores, N, rA, rb, rx0, nswp_,
    eps_, rmax_, max_full_, kickrank_, kick2_, local_iterations_, resets_,
    verbose_, prec_);

  // Create a new state
  const auto x = linalg::TTState::create(
    linalg::TTEngine::Tensors(x_cores.cbegin(), x_cores.cend()),
    x0->get_label());

  // Update the linear system
  local_system->set_state(x);
}

} // namespace ttnte::solvers
