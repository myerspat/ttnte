#include "ttnte/linalg/eig.hpp"
#include "ttnte/linalg/gmres.hpp"
#include <chrono>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

namespace ttnte::linalg {

std::tuple<torch::Tensor, double> power(std::shared_ptr<Operator> T,
  std::shared_ptr<Operator> F, const std::optional<torch::Tensor>& psi0,
  const double& tol, const int64_t& maxiter,
  const std::optional<int64_t>& gpu_idx,
  const std::optional<py::function>& callback,
  const int64_t& callback_frequency, const bool& verbose,
  const LinearSolverOptions& lsoptions)
{
  // Get size of system
  auto product = [](const std::vector<int64_t>& vec) {
    return std::reduce(vec.begin(), vec.end(), 1, std::multiplies<int64_t>());
  };
  int64_t n = product(T->input_shape());

  // Data types and device
  const auto& dtype = T->dtype();
  const auto& device = T->device();

  // Get initial eigenvector
  torch::Tensor psi =
    (psi0.has_value())
      ? psi0.value().clone().reshape({-1, 1})
      : torch::ones({n, 1}, torch::TensorOptions().dtype(dtype).device(device));

  // Normalize
  psi /= psi.norm(2);

  if (F->dtype() != dtype || F->device() != device || psi.dtype() != dtype ||
      psi.device() != device) {
    throw std::runtime_error(
      "T, F, and x0 should be on the same device with the same data type");
  }

  // Check system is square
  if (n != product(T->output_shape()) || n != product(F->input_shape()) ||
      n != product(F->output_shape())) {
    throw std::runtime_error(
      "The eigenvalue problem should be square with T and F matching in shape");
  }
  if (psi.size(0) != n) {
    throw std::runtime_error("x0 should have the same size as the input of T.");
  }

  // Send data to device
  if (gpu_idx.has_value()) {
    if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
      // Get device
      torch::Device device(torch::kCUDA, gpu_idx.value());

      // Send to device
      T->cuda(gpu_idx.value());
      F->cuda(gpu_idx.value());
      psi = psi.to(device);
    } else {
      std::cout << "WARNING: CUDA not available" << std::endl;
    }
  }

  // Calculate initial eigenvalue using Rayleigh quotient
  double k = (psi.t().matmul(F->apply(psi)) / psi.t().matmul(T->apply(psi)))
               .cpu()
               .item<double>();

  // Get total fission
  double ft = F->apply(psi).sum().cpu().item<double>();

  // Begin iteration
  int64_t i = 0;
  double error = std::numeric_limits<double>::max();
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose) {
    std::cout << "Running power iteration on "
              << ((gpu_idx.has_value())
                     ? "GPU " + std::to_string(gpu_idx.value())
                     : "CPU")
              << std::endl;
  }

  do {
    // Run GMRES
    const torch::Tensor psii = std::get<0>(gmres(T, (1 / k) * F->apply(psi),
      psi, lsoptions.gpu_idx, lsoptions.tol, lsoptions.atol, lsoptions.restart,
      lsoptions.maxiter, lsoptions.solve_method, lsoptions.callback,
      lsoptions.callback_frequency, lsoptions.verbose));

    // Calculate relative L2 error
    error = ((psii - psi).norm(2) / (psi).norm(2)).item<double>();
    psi = psii;

    // Calculate total fission source
    double fti = F->apply(psi).sum().item<double>();

    // Update the eigenvalue
    k *= fti / ft;
    ft = fti;

    // Flip sign if needed
    if (ft < 0 && k > 0) {
      psi *= -1;
      ft *= -1;
    }

    // Print progress
    if (i % callback_frequency == 0) {
      if (verbose) {
        std::cout << "-- (" << i << "): k = " << std::fixed
                  << std::setprecision(6) << k
                  << ", Angular Flux L2-Error = " << std::fixed
                  << std::setprecision(12) << error
                  << ", Elapsed Time = " << std::fixed << std::setprecision(3)
                  << static_cast<double>(
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count()) *
                       1e-3
                  << " s" << std::endl;
      }
      if (callback.has_value()) {
        callback.value()(i, error, psi);
      }
    }
    i++;

  } while (error >= tol && i < maxiter);

  // Move data back to CPU
  if (gpu_idx.has_value()) {
    T->cpu();
    F->cpu();
    psi = psi.cpu();
  }

  // Print convergence
  if (verbose) {
    std::cout << "-- " << ((error < tol) ? "Converged!" : "Failed to Converge!")
              << std::endl;
    ;
  }

  return {psi, k};
}

} // namespace ttnte::linalg
