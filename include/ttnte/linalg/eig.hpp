#pragma once

#include "ttnte/linalg/operator.hpp"
#include <optional>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;

namespace ttnte::linalg {

struct LinearSolverOptions {
  // Options
  double tol = 1e-10;
  double atol = 0.0;
  int64_t restart = 100;
  int64_t maxiter = 20;
  std::string solve_method = "batched";
  std::optional<py::function> callback = std::nullopt;
  std::optional<int64_t> gpu_idx = std::nullopt;
  int64_t callback_frequency = 1;
  bool verbose = false;
};

// Method for solving k-eigenvalue equation
std::tuple<torch::Tensor, double> power(std::shared_ptr<Operator> T,
  std::shared_ptr<Operator> F,
  const std::optional<torch::Tensor>& psi0 = std::nullopt,
  const double& tol = 1e-8, const int64_t& maxiter = 100,
  const std::optional<int64_t>& gpu_idx = std::nullopt,
  const std::optional<py::function>& callback = std::nullopt,
  const int64_t& callback_frequency = 1, const bool& verbose = true,
  const LinearSolverOptions& lsoptions = LinearSolverOptions());

} // namespace ttnte::linalg
