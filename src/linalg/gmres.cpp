#include "ttnte/linalg/gmres.hpp"
#include <c10/core/Device.h>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <torch/extension.h>

using namespace torch::indexing;

namespace ttnte::linalg {

LinearSystem::LinearSystem(const std::shared_ptr<Operator>& A,
  const torch::Tensor& b, const std::optional<torch::Tensor>& x0)
  : A(A), b(b.reshape({-1, 1}).contiguous()),
    x((x0.has_value())
        ? x0.value().clone().reshape({-1, 1}).contiguous()
        : torch::zeros_like(b, b.options()).reshape({-1, 1}).contiguous())
{
  // Get length expected by A
  auto product = [](const std::vector<int64_t>& vec) {
    return std::reduce(vec.begin(), vec.end(), 1, std::multiplies<int64_t>());
  };

  // Check data types
  // TODO: Add check on A as well
  if (b.dtype() != torch::kFloat64 && x.dtype() != torch::kFloat64) {
    throw std::runtime_error("Only double precision is supported for GMRES");
  }

  // Check shape of system
  if (product(A->input_shape()) != b.size(0) || b.size(0) != x.size(0)) {
    throw std::runtime_error(
      "Shapes of linear system Ax = b must be consistent");
  }

  // Check A is square
  if (product(A->input_shape()) != product(A->output_shape())) {
    throw std::runtime_error("A must be square");
  }
}

double safe_normalize(torch::Tensor& x, const std::optional<double>& thresh)
{
  // Normalize x
  double norm = x.norm().item<double>();

  // Check if norm is above threshold
  if (norm > ((!thresh.has_value()) ? std::numeric_limits<double>::epsilon()
                                    : thresh.value())) {
    x = 1 / norm * x;
  } else {
    x = torch::zeros_like(x);
    norm = 0.0;
  }
  return norm;
}

bool kth_arnoldi_iteration(const int64_t& k, std::shared_ptr<Operator> A,
  torch::Tensor& V, torch::Tensor& H)
{
  // Compute next candidate vector
  torch::Tensor v = A->apply(V.index({Slice(), k})).reshape({-1, 1});

  // Calculate norm of candidate
  double vnorm = v.norm().item<double>();
  if (vnorm < std::numeric_limits<double>::epsilon())
    vnorm = 0.0;

  // Run classical Gram-Schmidt
  auto h = iterative_classical_gram_schmidt(V, v, vnorm, 2);

  // Calculate updated tolerance
  double tol = std::numeric_limits<double>::epsilon() * vnorm;
  vnorm = safe_normalize(v);

  V.index_put_({Slice(), k + 1}, v.flatten());
  h.index_put_({k + 1}, vnorm);
  H.index_put_({k, Slice()}, h.flatten());

  // Check for a breakdown
  return vnorm == 0;
}

torch::Tensor iterative_classical_gram_schmidt(const torch::Tensor& Q,
  torch::Tensor& q, const double& qnorm, const int64_t& maxiter)
{
  assert(Q.dim() == 2);

  // Setup
  int64_t k = 0;
  double sqrt2_inv = 1 / std::sqrt(2.0);
  torch::Tensor r = torch::zeros({Q.size(1), 1}, Q.options());
  double qnorm_scaled = qnorm * sqrt2_inv;
  double rnorm = std::numeric_limits<double>::max();

  // Begin iteration
  do {
    {
      // Project, subtract, and accumulate
      torch::Tensor h = Q.t().matmul(q);
      q -= Q.matmul(h);
      r += h;
    }

    // Compute safe norm of q
    double qnorm_ = q.norm().item<double>();
    if (qnorm_ < std::numeric_limits<double>::epsilon())
      qnorm_ = 0.0;
    qnorm_scaled = qnorm_ * sqrt2_inv;

    // Compute safe norm of r
    rnorm = r.norm().item<double>();
    if (rnorm < std::numeric_limits<double>::epsilon())
      rnorm = 0.0;

    // Increment
    k++;
  } while (rnorm < qnorm_scaled && k < maxiter);

  return r;
}

// =========================================================================
// General GMRES method
std::tuple<torch::Tensor, torch::Tensor> gmres(std::shared_ptr<Operator> A,
  torch::Tensor& b, const std::optional<torch::Tensor>& x0,
  const std::optional<int64_t>& gpu_idx, double tol, double atol,
  int64_t restart, std::optional<int64_t> maxiter,
  const std::string& solve_method, std::optional<py::function> callback,
  const int64_t& callback_frequency, const bool& verbose)
{
  // Send to GPU
  if (gpu_idx.has_value()) {
    if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
      // Get device
      torch::Device device(torch::kCUDA, gpu_idx.value());

      // Send to device
      A->cuda(gpu_idx.value());
      b = b.to(device);
    } else {
      std::cout << "WARNING: CUDA not available" << std::endl;
    }
  }

  // Create linear system
  struct LinearSystem system(A, b, x0);

  // Get maximum number of iterations
  int64_t n_ = b.size(0);
  maxiter = (maxiter.has_value()) ? maxiter.value() : (10 * n_);

  // Get restart size
  restart = std::min(restart, n_);

  // Get norms
  // TODO: Implement preconditioning
  double bnorm = system.b.norm().item<double>();
  double bnorm_inv = 1 / bnorm;
  atol = std::max(tol * bnorm, atol);
  double ptol = bnorm * std::min(1.0, atol / bnorm);

  // Calculate residual
  torch::Tensor residual = system.calc_residual();
  torch::Tensor residual_log = torch::empty({maxiter.value() + 1});
  double rnorm = safe_normalize(residual);
  residual_log.index_put_({0}, rnorm + 1);

  // Iteration
  int64_t i = 0;

  // Function to apply at each iteration
  std::function<void(
    LinearSystem&, torch::Tensor&, double&, const double&, const int64_t&)>
    gmres_func;

  if (solve_method == "batched") {
    gmres_func = &gmres_batched;
  } else if (solve_method == "incremental") {
    gmres_func = &gmres_incremental;
  } else {
    throw std::runtime_error(
      "solve_method must be either 'batched' or 'incremental'");
  }

  // Notify user
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose) {
    std::cout << "Running " + solve_method + " GMRES on "
              << ((gpu_idx.has_value())
                     ? "GPU " + std::to_string(gpu_idx.value())
                     : "CPU")
              << std::endl;
  }

  // Begin iteration
  do {
    gmres_func(system, residual, rnorm, ptol, restart);
    i++;

    // Track residual
    residual_log.index_put_({i}, rnorm);

    // Run callback and prints
    if (i % callback_frequency == 0) {
      if (verbose) {
        printf("-- (%d): |r| = %.12f, |r|/|b| = %.12f, Elapsed Time = %.3f s\n",
          static_cast<int>(i), rnorm, bnorm_inv * rnorm,
          static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - start)
              .count()) *
            1e-3);
      }
      if (callback.has_value()) {
        callback.value()(i, rnorm, system.x);
      }
    }
  } while (rnorm > atol && i < maxiter.value());

  if (gpu_idx.has_value()) {
    // Get CPU
    torch::Device device(torch::kCPU);

    // Remove system from GPU
    system.A->cpu();
    system.b = b.to(device);
    system.x = system.x.to(device);
    residual = residual.to(device);
  }

  return {system.x, residual_log.narrow(0, 0, i + 1)};
}

// =========================================================================
// Batched GMRES helper methods
void gmres_batched(LinearSystem& system, torch::Tensor& residual, double& rnorm,
  const double& ptol, const int64_t& restart)
{
  assert(residual.size(1) == 1 && residual.ndimensional() == 2);

  // Initialize Krylov basis
  torch::Tensor V = residual.clone();
  V = torch::constant_pad_nd(V, {0, restart});

  // Create initial upper Hessenberg matrix
  torch::Tensor H = torch::eye(restart, restart + 1, V.options());

  // Iteration information
  bool breakdown = false;
  int64_t k = 0;

  // Run iteration
  do {
    breakdown = kth_arnoldi_iteration(k, system.A, V, H);
    k++;
  } while (k < restart && !breakdown);

  // Initialize beta
  torch::Tensor beta = torch::zeros({restart + 1, 1}, H.options());
  beta.index_put_({0, 0}, rnorm);

  // Solve normal equation and apply it to x
  system.x += V.index({Slice(), Slice(0, -1)})
                .matmul(torch::cholesky_solve(
                  H.matmul(beta), torch::linalg_cholesky(H.matmul(H.t()))));

  // Update residual and its norm
  residual = system.calc_residual();
  rnorm = safe_normalize(residual);
}

// =========================================================================
// Incremental GMRES helper methods
void gmres_incremental(LinearSystem& system, torch::Tensor& residual,
  double& rnorm, const double& ptol, const int64_t& restart)
{
  assert(residual.size(1) == 1 && residual.ndimensional() == 2);

  // Initialize Krylov basis
  torch::Tensor V = residual.clone();
  V = torch::constant_pad_nd(V, {0, restart});

  // Create initial upper rectangular matrix
  torch::Tensor R = torch::eye(restart, restart + 1, V.options());

  // Givens and beta
  torch::Tensor givens = torch::zeros({restart, 2}, V.options());
  torch::Tensor beta = torch::zeros({restart + 1}, V.options());
  beta.index_put_({0}, rnorm);

  // Iteration data
  int64_t k = 0;
  double presid = 1.0;
  bool breakdown = false;

  do {
    // Run kth arnoldi iteration
    bool breakdown = kth_arnoldi_iteration(k, system.A, V, R);

    // Get view into kth row of R
    torch::Tensor R_row = R.index({k, Slice()});

    // Apply rotation
    for (int64_t i = 0; i < k; i++) {
      rotate_vectors(i, R_row, givens.index({i, Slice()}));
    }

    // Calculate Givens factors and apply last rotation to R
    givens.index_put_(
      {k, Slice()}, givens_rotation(R_row.index({Slice(k, k + 2)})));
    rotate_vectors(k, R_row, givens.index({k, Slice()}));

    // Rotate beta vector and update residual norm
    rotate_vectors(k, beta, givens.index({k, Slice()}));
    presid =
      abs(beta[k + 1].item<double>() * givens.index({k, 1}).item<double>());

    k++;
  } while (presid > ptol * 1e-2 && k < restart && !breakdown);

  // Solve system and update x
  system.x += V.index({Slice(), Slice(0, -1)})
                .matmul(torch::linalg_solve_triangular(
                  R.index({Slice(), Slice(0, -1)}).t(),
                  beta.index({Slice(0, -1)}).reshape({-1, 1}), true));

  // Update residual and its norm
  residual = system.calc_residual();
  rnorm = safe_normalize(residual);
}

void rotate_vectors(const int64_t& i, torch::Tensor H, torch::Tensor givens)
{
  assert(givens.ndimensional() == 1 && givens.size(0) == 2);
  assert(H.ndimensional() == 1);

  double x = givens[0].item<double>() * H[i].item<double>() -
             givens[1].item<double>() * H[i + 1].item<double>();
  double y = givens[1].item<double>() * H[i].item<double>() +
             givens[0].item<double>() * H[i + 1].item<double>();

  H.index_put_({Slice(i, i + 2)}, torch::tensor({x, y}, H.options()));
}

torch::Tensor givens_rotation(const torch::Tensor& factors)
{
  torch::Tensor new_factors = torch::tensor({1, 0});

  // If second factor is zero
  if (abs(factors.index({1}).item<double>()) == 0) {
    return torch::tensor({1.0, 0.0}, factors.options());
  }

  // If first factor < second factor
  if (abs(factors.index({0}).item<double>()) <
      abs(factors.index({1}).item<double>())) {
    double t =
      -(factors.index({0}).item<double>() / factors.index({1}).item<double>());
    double r = 1 / std::sqrt(1.0 + abs(t) * abs(t));

    return torch::tensor({r * t, r}, factors.options());
  }

  // If first factor > second factor
  double t =
    -(factors.index({1}).item<double>() / factors.index({0}).item<double>());
  double r = 1 / std::sqrt(1.0 + abs(t) * abs(t));

  return torch::tensor({r, r * t}, factors.options());
}

} // namespace ttnte::linalg
