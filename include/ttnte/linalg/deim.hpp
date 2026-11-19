#pragma once

#include <cstdint>
#include <optional>
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
/// Rows already chosen are excluded, so the rows are distinct.
/// @return The r selected rows, in selection order.
/// @throws std::runtime_error If V is not 2-D with 1 <= r <= n, or V is
/// (numerically) rank deficient: a column's largest interpolation residual is
/// at rounding level (max(n, r) eps times its norm), which would make the
/// interpolation matrix V(I, :) singular.
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
/// @throws std::runtime_error On mismatched shapes, r > n or a nonpositive
/// weight.
std::pair<torch::Tensor, torch::Tensor> weighted_orthonormal(
  const torch::Tensor& L, const torch::Tensor& w);

/// @brief The interpolation's Lebesgue constant: the norm of the projector
/// f -> V V(I, :)^{-1} f(I), the factor by which interpolation at the rows I
/// can exceed the best approximation in span(V). Unweighted it is
/// ||V_o(I, :)^+||_2, V_o an orthonormal basis of span(V); with weights w it
/// is measured in ||g||_w = ||diag(sqrt w) g||_2 (quadrature-weighted, as Han
/// et al.'s V and the moments are) and V_o spans diag(sqrt w) V.
/// @param V (n, r) basis, r <= n.
/// @param I Selected rows, exactly r (the square interpolation of DEIM).
/// @param w Optional (n) positive weights.
/// @return The constant (infinite if V is numerically rank deficient or
/// V_o(I, :) is singular).
/// @throws std::runtime_error On mismatched shapes or a nonpositive weight.
double lebesgue_constant(const torch::Tensor& V, const std::vector<int64_t>& I,
  const std::optional<torch::Tensor>& w = std::nullopt);

} // namespace ttnte::linalg
