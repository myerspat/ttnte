#include "ttnte/sweep/iterative_solver.hpp"
#include "ttnte/linalg/krylov.hpp"

namespace ttnte::sweep {

IterativeSolver::IterativeSolver(const PatchDiscretization& disc, double tol,
  int64_t restart, int64_t max_iter)
  : op_(SumFactorization::create(disc)), ndim_(disc.ndim()), tol_(tol),
    restart_(restart), max_iter_(max_iter)
{
  torch::Tensor metric;
  twin_ = KroneckerSolver::parametric_twin(disc, metric, measure_);
  metric_.assign(
    metric.data_ptr<double>(), metric.data_ptr<double>() + metric.numel());
}

bool IterativeSolver::solve(
  const double* omega, double sigma, double* b, Workspace& work) const
{
  const int64_t N = op_->num_dofs();
  // Preconditioner: the twin with velocity metric * Omega
  double beta[3] = {0.0, 0.0, 0.0};
  for (int64_t k = 0; k < ndim_; k++)
    for (int64_t i = 0; i < ndim_; i++)
      beta[k] += metric_[k * ndim_ + i] * omega[i];
  const double sigma_twin = measure_ * sigma;
  auto precondition = [&](const double* in, double* out) {
    std::copy(in, in + N, out);
    twin_->solve(beta, sigma_twin, out, work.twin);
  };

  // Right-preconditioned GMRES: L P^{-1} y = b, x = P^{-1} y
  work.t.resize(N);
  auto apply = [&](const double* in, double* out) {
    precondition(in, work.t.data());
    op_->apply(omega, sigma, work.t.data(), out, work.apply);
  };
  work.rhs.assign(b, b + N);
  work.y.assign(N, 0.0);
  auto& y = work.y;
  auto result = linalg::krylov::gmres(
    apply, work.rhs, y, restart_, tol_, max_iter_, work.gmres);
  solves_++;
  iterations_ += result.num_applies;
  if (!result.converged) {
    failures_++;
    return false;
  }
  precondition(y.data(), b);
  return true;
}

} // namespace ttnte::sweep
