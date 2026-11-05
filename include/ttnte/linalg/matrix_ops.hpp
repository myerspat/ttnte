#pragma once

#include <limits>
#include <torch/extension.h>
#include <tuple>

namespace ttnte::linalg {

/// @brief A is treated as tall (or wide) when its long dimension is at least
/// this many times its short one; the Gram paths below need such shapes.
inline constexpr int64_t SKINNY_RATIO = 10;

/// @brief Compute a truncated singular value decomposition (SVD).
/// @param A The matrix to perform a truncated SVD operation.
/// @param delta The error on the Frobenius norm of the matrix for truncation.
/// @param max_rank The maximum allowed rank of the final SVD split.
/// @param with_normalize Whether to normalize `delta` with the Frobenius
/// norm of A.
/// @return A tuple of U, S, Vh, delta (potentially normalized), and the
/// remaining delta.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double>
truncated_svd(const torch::Tensor& A, double delta = 0,
  int64_t max_rank = std::numeric_limits<int64_t>::max(),
  bool with_normalize = false);

/// @brief Gram matrix of a tall A (A^T A) or a wide A (A A^T), summed
/// over blocks of the long dimension with one batched product. A plain
/// product with such a long inner dimension runs at a few GB/s and barely
/// speeds up with threads; the blocked one scales (Big Ten-sized
/// 1.27M x 16 at 32 threads: 50 to 2 ms).
torch::Tensor blocked_gram(const torch::Tensor& A);

/// @brief Truncated SVD of a tall or wide matrix through its Gram matrix:
/// the symmetric eigendecomposition of A^T A (tall) or A A^T (wide), one
/// matrix product of the long dimension instead of a Householder QR. The
/// singular values are accurate to about sqrt(machine epsilon) times the
/// largest, and the long factor (U = A W / sigma, or its transpose) is
/// orthonormal only to about eps (sigma_1 / sigma_r)^2 (about 1e-5 at the
/// floor), so below sigma_r / sigma_1 = 1e-3 it is re-orthogonalized (one
/// CholeskyQR pass, two more passes over it) unless `orthonormal` is false.
/// It falls back to truncated_svd() when the truncation threshold
/// is below gram_floor times the Frobenius norm, or when A is neither
/// SKINNY_RATIO times taller than wide nor wider than tall.
/// @param A The matrix to perform a truncated SVD operation.
/// @param delta The error on the Frobenius norm of the matrix for truncation.
/// @param max_rank The maximum allowed rank of the final SVD split.
/// @param with_normalize Whether to normalize `delta` with the Frobenius
/// norm of A.
/// @param gram_floor Smallest relative threshold delta / |A|_F handled
/// through the Gram matrix.
/// @param orthonormal Re-orthogonalize the long factor when needed (false
/// for callers that only need A ~ U S Vh, e.g. a cross fitting its cores
/// by least squares).
/// @return As truncated_svd().
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double>
gram_truncated_svd(const torch::Tensor& A, double delta = 0,
  int64_t max_rank = std::numeric_limits<int64_t>::max(),
  bool with_normalize = false, double gram_floor = 1e-7,
  bool orthonormal = true);

} // namespace ttnte::linalg
