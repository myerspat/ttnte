#pragma once

#include "ttnte/linalg/operator.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace ttnte::linalg {

struct LinearSystem {
  // Data
  std::shared_ptr<Operator> A;
  torch::Tensor b;
  torch::Tensor x;

  // Constructor
  LinearSystem(const std::shared_ptr<Operator>& A, const torch::Tensor& b,
    const std::optional<torch::Tensor>& x0 = std::nullopt);

  // Methods
  torch::Tensor calc_residual() const { return b - A->apply(x); }
};

/* The following implementations are taken from JAX's version of Scipy:
 * https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/sparse/linalg.py.
 * Citation:
 * @software{jax2018github,
 *  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James
 *            Johnson and Chris Leary and Dougal Maclaurin and George Necula and
 *            Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and
 *            Qiao Zhang},
 *  title ={{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
 *  url = {http://github.com/jax-ml/jax},
 *  version = {0.3.13},
 *  year = {2018},
 * }
 */

double safe_normalize(
  torch::Tensor& x, const std::optional<double>& thresh = std::nullopt);

bool kth_arnoldi_iteration(const int64_t& k, std::shared_ptr<Operator> A,
  torch::Tensor& V, torch::Tensor& H);

torch::Tensor iterative_classical_gram_schmidt(const torch::Tensor& Q,
  torch::Tensor& q, const double& qnorm, const int64_t& maxiter = 2);

// =========================================================================
// General GMRES method
std::tuple<torch::Tensor, torch::Tensor> gmres(std::shared_ptr<Operator> A,
  torch::Tensor b, const std::optional<torch::Tensor>& x0 = std::nullopt,
  const std::optional<int64_t>& gpu_idx = std::nullopt, double tol = 1e-5,
  double atol = 0.0, int64_t restart = 20,
  std::optional<int64_t> maxiter = std::nullopt,
  const std::string& solve_method = "batched",
  std::optional<py::function> callback = std::nullopt,
  const int64_t& callback_frequency = 1, const bool& verbose = true);

// =========================================================================
// Batched GMRES helper methods
void gmres_batched(LinearSystem& system, torch::Tensor& residual, double& rnorm,
  const double& ptol, const int64_t& restart);

// =========================================================================
// Incremental GMRES helper methods
void gmres_incremental(LinearSystem& system, torch::Tensor& residual,
  double& rnorm, const double& ptol, const int64_t& restart);

void rotate_vectors(const int64_t& i, torch::Tensor H, torch::Tensor givens);

torch::Tensor givens_rotation(const torch::Tensor& factors);

} // namespace ttnte::linalg
