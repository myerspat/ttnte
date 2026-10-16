#pragma once

#include <cstdint>
#include <functional>
#include <torch/extension.h>
#include <vector>

namespace ttnte::cross {

/// @brief Columns of R chosen by Householder QR with column pivoting until
/// the unpicked part is below eps ||R||_F (at least one column).
/// @param R (k, n) matrix, k small.
/// @param eps Relative tolerance.
/// @return Column indices in pivot order.
std::vector<int64_t> pivot_columns(const torch::Tensor& R, double eps);

/// @brief Gaussian sketch matrix of one direction's rows: (k, n) standard
/// normals generated from the direction index alone, so every partition of
/// the rows builds the same sketch without communication.
torch::Tensor direction_sketch(int64_t direction, int64_t k, int64_t n);

/// @brief Whole blocks of rows (rows sharing a key) chosen until B[rows]
/// has sigma_min / sigma_max >= cond. The next block maximizes the residual
/// energy of its rows outside the resolved row space, or is a cheaper one
/// (lower cost) within `near` of the best. Blocks whose key is in `start`
/// are always taken.
/// @param B (m, r) basis evaluated at the candidate rows.
/// @param keys (m) block key of every row.
/// @param cost Cost of a block, by key (fibers it still needs).
/// @param start Keys that are always taken.
/// @return The chosen rows, ascending.
std::vector<int64_t> select_blocks(const torch::Tensor& B,
  const std::vector<int64_t>& keys, const std::function<double(int64_t)>& cost,
  const std::vector<int64_t>& start, double cond = 1e-2, double near = 0.25);

} // namespace ttnte::cross
