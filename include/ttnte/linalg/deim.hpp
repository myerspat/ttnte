#pragma once

#include <cstdint>
#include <torch/extension.h>
#include <utility>
#include <vector>

namespace ttnte::linalg {

// =============================================================================
// Discrete empirical interpolation: selecting rows of a basis
// =============================================================================

/// @brief DEIM row selection (Chaturantabut & Sorensen; Han et al. 2026,
/// Alg. B.1): the first row maximizes |V(:, 0)|, each next one the
/// interpolation residual of the next column on the rows chosen so far.
/// @param V (n, r) basis, r <= n.
/// @return The r selected rows, in selection order.
/// @throws std::runtime_error If V is not 2-D with 1 <= r <= n, or the
/// interpolation matrix of the chosen rows becomes singular.
std::vector<int64_t> deim_select(const torch::Tensor& V);

/// @brief Q-DEIM row selection (Drmac & Gugercin 2016): the first r column
/// pivots of a column-pivoted QR of V^T (LAPACK dgeqp3, single-threaded).
/// @param V (n, r) basis, r <= n.
/// @return The r selected rows, in pivot order.
/// @throws std::runtime_error If V is not 2-D with 1 <= r <= n, or LAPACK
/// fails.
std::vector<int64_t> qdeim_select(const torch::Tensor& V);

/// @brief A quadrature-weighted orthonormal basis of span(L) (Han et al.
/// 2026, eq. 3.5): L = V R with V^T diag(w) V = I, from the QR of
/// diag(sqrt(w)) L.
/// @param L (n, r) coefficients, row per quadrature point.
/// @param w (n) positive weights.
/// @return V (n, r) and R (r, r) (upper triangular).
/// @throws std::runtime_error On mismatched shapes or a nonpositive weight.
std::pair<torch::Tensor, torch::Tensor> weighted_orthonormal(
  const torch::Tensor& L, const torch::Tensor& w);

/// @brief The interpolation's Lebesgue constant ||V_o(I, :)^+||_2, V_o an
/// orthonormal basis of span(V): the factor by which interpolation at the
/// rows I can exceed the best approximation in span(V).
/// @param V (n, r) basis.
/// @param I Selected rows (at least r).
/// @return The constant (infinite if V_o(I, :) is rank deficient).
double lebesgue_constant(const torch::Tensor& V, const std::vector<int64_t>& I);

} // namespace ttnte::linalg
