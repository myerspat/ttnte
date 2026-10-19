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

/// @brief Truncated SVD of a tall or wide matrix through its Gram matrix:
/// the symmetric eigendecomposition of A^T A (tall) or A A^T (wide), one
/// matrix product of the long dimension instead of a Householder QR. The
/// singular values are accurate to about sqrt(machine epsilon) times the
/// largest, so it falls back to truncated_svd() when the truncation
/// threshold is below gram_floor times the Frobenius norm, or when A is
/// neither 10 times taller than wide nor 10 times wider than tall.
/// @param A The matrix to perform a truncated SVD operation.
/// @param delta The error on the Frobenius norm of the matrix for truncation.
/// @param max_rank The maximum allowed rank of the final SVD split.
/// @param with_normalize Whether to normalize `delta` with the Frobenius
/// norm of A.
/// @param gram_floor Smallest relative threshold delta / |A|_F handled
/// through the Gram matrix.
/// @return As truncated_svd().
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double>
gram_truncated_svd(const torch::Tensor& A, double delta = 0,
  int64_t max_rank = std::numeric_limits<int64_t>::max(),
  bool with_normalize = false, double gram_floor = 1e-7);

} // namespace ttnte::linalg
