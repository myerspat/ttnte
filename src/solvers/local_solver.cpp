#include "ttnte/solvers/local_solver.hpp"

namespace ttnte::solvers {

// =================================================================
// Public methods
std::tuple<linalg::Operator, linalg::State, linalg::State>
LocalSolver::presolve(const linalg::LinearSystem::Ptr& sys) const
{
  // Get the operators for the linear system
  linalg::Operator A = sys->get_interior_op();
  linalg::State x0 = sys->get_state();

  // Accumulate boundary contributions into a separate state so that
  // combining with the source uses the binary operator+ (which allocates a
  // fresh StateData), avoiding aliasing with EigenSource::state_ through the
  // shallow-copy State handle.
  linalg::State boundary_sum;
  bool has_boundary = false;
  for (auto& coupling : sys->get_couplings()) {
    if (coupling.recv_buffer.defined()) {
      if (boundary_sum.defined()) {
        boundary_sum += coupling.recv_buffer;
      } else {
        boundary_sum = std::move(coupling.recv_buffer);
      }
      coupling.recv_buffer = linalg::State();
      has_boundary = true;
    }
  }

  // Build the RHS: fission/fixed source + boundary.
  // When boundary is present, use binary operator+ so the result owns fresh
  // StateData — this prevents aliasing through the shallow-copy State handle
  // from corrupting EigenSource::state_ via operator+=.
  linalg::State b;
  const auto* src = sys->get_source().get();
  if (src && src->get_state().defined()) {
    if (has_boundary) {
      b = src->get_state() + boundary_sum;
      b.round_(get_eps(), get_max_rank());
    } else {
      b = src->get_state();
    }
  } else if (has_boundary) {
    b = std::move(boundary_sum);
    b.round_(get_eps(), get_max_rank());
  }

  return std::make_tuple(std::move(A), std::move(b), std::move(x0));
}

void LocalSolver::postsolve(
  const linalg::LinearSystem::Ptr& sys, const linalg::State& x) const
{
  const auto& x0 = sys->get_state();

  // Compute per-coupling boundary convergence error: compare the face of x
  // with the face of x0 along each internal boundary dimension. A rough
  // rounding is applied to the diff to keep its rank manageable — only a
  // convergence indicator is needed, not a precise residual.
  for (auto& coupling : sys->get_couplings()) {
    const size_t bdim = static_cast<size_t>(x.ndimension()) -
                        coupling.connection.mapping.flip.size() - 2 +
                        coupling.dim;
    auto face_new = x.narrow(bdim, coupling.is_upper ? -1 : 0, 1);
    auto face_old = x0.narrow(bdim, coupling.is_upper ? -1 : 0, 1);
    linalg::State diff = face_new - face_old;
    diff.round_(get_eps(), get_max_rank());
    const double n_diff = diff.norm();
    const double n_old = face_old.norm();
    coupling.sq_diff = n_diff * n_diff / 2.0;
    coupling.sq_prev = n_old * n_old / 2.0;
  }

  // Update the linear system
  sys->set_state(std::move(x));
}

} // namespace ttnte::solvers
