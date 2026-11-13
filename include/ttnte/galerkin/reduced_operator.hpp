#pragma once

#include <cstdint>
#include <memory>
#include <torch/extension.h>
#include <vector>

namespace ttnte::galerkin {

/// @brief A transport operator projected onto a growing basis, one reduced
/// system per direction (Guo & Peng, 2026, eq. 3.4): what a greedy Galerkin
/// sampler needs from the discretization.
///
/// A basis vector is one flux over the sampled unit's other indices (for the
/// sweep's ProjectedTransport: a group set's groups and every local unit,
/// in the fiber layout (num_groups, num_space), unweighted). With X the
/// basis, the reduced solution of direction d is c_d solving
///   X^T A_d X c_d = X^T b_d,
/// psi_d ~ X c_d, and its residual b_d - A_d X c_d is what the greedy step
/// scores. Norms are the sampler's (sampling::FiberStore::residual_norms()),
/// so that the weighting has a single owner.
class ReducedOperator {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<ReducedOperator>;

  virtual ~ReducedOperator() = default;

  // =================================================================
  // Public methods
  /// @brief Drop the basis (a new transport solve: new sources and traces).
  virtual void reset() = 0;

  /// @brief Append basis vectors and border every projection.
  /// @param X (p, num_groups, num_space) new basis vectors, unweighted. The
  /// existing vectors are never changed (the projections are bordered).
  virtual void extend(const torch::Tensor& X) = 0;

  /// @brief Solve the reduced systems of `directions`, in parallel over
  /// directions (at::parallel_for: the caller's thread count).
  /// @return (D, r) the coefficients c_d, row per direction.
  /// @throws std::runtime_error If a reduced matrix is singular.
  virtual torch::Tensor solve(const std::vector<int64_t>& directions) = 0;

  /// @brief Residuals b_d - A_d X c_d over the whole domain, in parallel
  /// over directions.
  /// @param C (D, r) coefficients, row per direction.
  /// @return (D, num_groups, num_space) residuals (integrals against the
  /// test functions, unweighted).
  virtual torch::Tensor residuals(
    const std::vector<int64_t>& directions, const torch::Tensor& C) const = 0;

  // =================================================================
  // Public getters
  /// @return Number of basis vectors.
  virtual int64_t rank() const noexcept = 0;
  virtual int64_t num_directions() const noexcept = 0;
  /// @return (r, num_groups, num_space) the basis.
  virtual torch::Tensor basis() const = 0;
};

} // namespace ttnte::galerkin
