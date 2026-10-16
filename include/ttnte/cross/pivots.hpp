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

/// @brief Blocks chosen by their Gram matrices G_u = B_u^T B_u until the
/// chosen rows have sigma_min / sigma_max >= cond: the next block maximizes
/// its residual energy tr(G_u) - <Pi, G_u> outside the resolved row space
/// (Pi its projector), or is a cheaper one within `near` of the best. Blocks
/// marked in `picked` on entry are always taken.
/// @param grams (nk, r, r) block Grams.
/// @param sizes (nk) rows per block.
/// @param cost (nk) cost of each block (fibers it still needs).
/// @param picked (nk) in: forced blocks; out: chosen blocks.
void select_block_grams(const torch::Tensor& grams,
  const std::vector<int64_t>& sizes, const std::vector<double>& cost,
  std::vector<uint8_t>& picked, double cond = 1e-2, double near = 0.25);

/// @brief select_block_grams() for blocks of one row each, by greedy pivoted
/// Gram-Schmidt: an orthonormal basis of the chosen rows is extended one
/// vector at a time and every row's residual updated by one product, so the
/// selection costs O(m r^2) instead of an eigendecomposition and a pass over
/// every block per step. The conditioning test is the same.
/// @param rows (m, r) the rows (one block each).
/// @param cost (m) cost of each row.
/// @param picked (m) in: forced rows; out: chosen rows.
void select_single_rows(const torch::Tensor& rows,
  const std::vector<double>& cost, std::vector<uint8_t>& picked,
  double cond = 1e-2, double near = 0.25);

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
