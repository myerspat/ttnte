#pragma once

#include <cstdint>
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
/// @param cond, near CrossConfig::select_cond and select_near.
void select_block_grams(const torch::Tensor& grams,
  const std::vector<int64_t>& sizes, const std::vector<double>& cost,
  std::vector<uint8_t>& picked, double cond, double near);

/// @brief select_block_grams() for blocks of one row each, by greedy pivoted
/// Gram-Schmidt: an orthonormal basis of the chosen rows is extended one
/// vector at a time and every row's residual updated by one product, so a
/// step costs O(m r) instead of a pass over every block's Gram. The
/// conditioning test is the same (an r x r eigenvalue problem per step once
/// r rows are chosen). Once the chosen rows span all r directions the
/// residuals are zero, and rows are scored as in select_block_grams() by
/// their energy in the directions with sigma_j <= cond sigma_max.
/// @param rows (m, r) the rows (one block each).
/// @param cost (m) cost of each row.
/// @param picked (m) in: forced rows; out: chosen rows.
/// @param cond, near CrossConfig::select_cond and select_near.
void select_single_rows(const torch::Tensor& rows,
  const std::vector<double>& cost, std::vector<uint8_t>& picked, double cond,
  double near);

} // namespace ttnte::cross
