#pragma once

#include "ttnte/utils/exception.hpp"
#include <cstdint>
#include <string>
#include <torch/extension.h>

namespace ttnte::linalg {

// =================================================================
// Batched drivers shared by the dense and banded LU: validate the storage,
// then run a per-matrix kernel over the batch.

/// @brief Factor every matrix of a (batch, n, width) contiguous CPU float64
/// tensor in place.
/// @param where Fully-qualified call site for error messages.
/// @param factor `int64_t factor(double* matrix, int64_t* pivots)`, returning
/// 0, or j + 1 for a zero pivot at column j.
/// @return The (batch, n) int64 pivots.
template<typename Factor>
torch::Tensor batched_factor(
  const char* where, torch::Tensor& a, int64_t min_width, Factor&& factor)
{
  if (a.dim() != 3 || a.scalar_type() != torch::kFloat64 ||
      !a.is_contiguous() || !a.device().is_cpu() || a.size(2) < min_width) {
    throw utils::runtime_error(where,
      "Expected a contiguous CPU float64 tensor of shape (batch, n, width) "
      "with width >= " +
        std::to_string(min_width));
  }
  const int64_t batch = a.size(0), n = a.size(1), width = a.size(2);
  auto ipiv = torch::empty({batch, n}, torch::kInt64);
  double* ap = a.data_ptr<double>();
  int64_t* pp = ipiv.data_ptr<int64_t>();
  for (int64_t k = 0; k < batch; k++) {
    const int64_t info = factor(ap + k * n * width, pp + k * n);
    if (info != 0) {
      throw utils::runtime_error(
        where, "Matrix " + std::to_string(k) +
                 " of the batch is singular (zero pivot at column " +
                 std::to_string(info - 1) + ")");
    }
  }
  return ipiv;
}

/// @brief Solve every factored system of a batch in place.
/// @param a (batch, n, width) factors from batched_factor().
/// @param ipiv (batch, n) pivots from batched_factor().
/// @param b (batch, n) right-hand sides, overwritten with the solutions.
/// @param solve `void solve(const double* matrix, const int64_t* pivots,
/// double* rhs)`.
template<typename Solve>
void batched_solve(const char* where, const torch::Tensor& a,
  const torch::Tensor& ipiv, torch::Tensor& b, Solve&& solve)
{
  if (a.dim() != 3 || a.scalar_type() != torch::kFloat64 ||
      !a.is_contiguous() || !a.device().is_cpu()) {
    throw utils::runtime_error(where,
      "Expected contiguous CPU float64 factors of shape (batch, n, width)");
  }
  const int64_t batch = a.size(0), n = a.size(1), width = a.size(2);
  if (ipiv.dim() != 2 || ipiv.scalar_type() != torch::kInt64 ||
      !ipiv.is_contiguous() || ipiv.size(0) != batch || ipiv.size(1) != n) {
    throw utils::runtime_error(
      where, "Expected contiguous int64 pivots of shape (batch, n)");
  }
  if (b.dim() != 2 || b.scalar_type() != torch::kFloat64 ||
      !b.is_contiguous() || b.size(0) != batch || b.size(1) != n) {
    throw utils::runtime_error(where,
      "Expected contiguous float64 right-hand sides of shape (batch, n)");
  }
  const int64_t* pp = ipiv.data_ptr<int64_t>();
  for (int64_t i = 0; i < batch * n; i++) {
    if (pp[i] < 0 || pp[i] >= n)
      throw utils::runtime_error(where, "Pivot index out of range");
  }
  const double* ap = a.data_ptr<double>();
  double* bp = b.data_ptr<double>();
  for (int64_t k = 0; k < batch; k++)
    solve(ap + k * n * width, pp + k * n, bp + k * n);
}

} // namespace ttnte::linalg
