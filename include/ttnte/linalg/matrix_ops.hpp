#pragma once

#include <limits>
#include <torch/extension.h>
#include <tuple>

namespace ttnte::linalg {

/// @brief Compute a truncated singular value decomposition (SVD).
/// @param A The matrix to perform a truncated SVD operation.
/// @param delta The error on the Frobenius norm of the matrix for truncation.
/// @param max_rank The maximum allowed rank of the final SVD split.
/// @param with_normalize Whether to normalize `delta` with `A.norm(2)`.
/// @return A tuple of U, S, Vh, delta (potentially normalized), and the
/// remaining delta.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double>
truncated_svd(const torch::Tensor& A, double delta = 0,
  int64_t max_rank = std::numeric_limits<int64_t>::max(),
  bool with_normalize = false);

} // namespace ttnte::linalg
