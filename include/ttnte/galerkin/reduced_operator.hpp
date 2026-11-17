#pragma once

#include "ttnte/utils/exception.hpp"
#include <cstdint>
#include <memory>
#include <torch/extension.h>
#include <utility>
#include <vector>

namespace ttnte::galerkin {

/// @brief A transport operator projected onto a growing basis, one reduced
/// system per sampling unit (Guo & Peng, 2026, eq. 3.4): what a greedy
/// Galerkin sampler needs from the discretization.
///
/// A sampling unit is a direction (its flux over every group of the set) or
/// a (direction, group) pair, u = d num_groups + g. A basis vector is one
/// flux over the unit's other indices: (basis_groups, num_space), with
/// basis_groups the set's groups for directions and 1 for pairs
/// (unweighted; for the sweep's ProjectedTransport in the fiber layout).
/// With X the basis, the reduced solution of unit u is c_u solving
///   X^T A_u X c_u = X^T b_u,
/// psi_u ~ X c_u, and its residual b_u - A_u X c_u is what the greedy step
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
  /// @param X (p, basis_groups, num_space) new basis vectors, unweighted.
  /// The existing vectors are never changed (the projections are bordered).
  virtual void extend(const torch::Tensor& X) = 0;

  /// @brief Solve the reduced systems of `units` (distinct), in parallel
  /// over units (at::parallel_for: the caller's thread count).
  /// @return (U, r) the coefficients c_u, row per unit.
  /// @throws std::runtime_error If a reduced matrix is singular.
  virtual torch::Tensor solve(const std::vector<int64_t>& units) = 0;

  /// @brief Residuals b_u - A_u X c_u over the whole domain, of a batch of
  /// units.
  /// @param C (U, r) coefficients, row per unit.
  /// @return (U, basis_groups, num_space) residuals (integrals against the
  /// test functions, unweighted).
  virtual torch::Tensor residuals(
    const std::vector<int64_t>& units, const torch::Tensor& C) const = 0;

  /// @brief The projected fixed source of the running transport solve: X^T
  /// of its source without the within-set scattering, per angular moment
  /// (Han et al.'s DEIM stage). Default: unsupported.
  /// @return (num_moments, r).
  /// @throws std::runtime_error If the operator does not project scattering.
  virtual torch::Tensor fixed_source() const
  {
    throw utils::runtime_error(
      "ttnte::galerkin::ReducedOperator::fixed_source", "Not supported");
  }

  /// @brief Every direction's coefficients psi_d = X L_d, coupled through
  /// the within-set scattering (Han et al. 2026, eq. 3.4), for a projected
  /// fixed source F. Default: unsupported.
  /// @return L (num_directions, r) and the moments' coefficients Lam (M, r).
  /// @throws std::runtime_error If the operator does not project scattering.
  virtual std::pair<torch::Tensor, torch::Tensor> angular_galerkin(
    const torch::Tensor& F)
  {
    (void)F;
    throw utils::runtime_error(
      "ttnte::galerkin::ReducedOperator::angular_galerkin", "Not supported");
  }

  // =================================================================
  // Public getters
  /// @return Number of basis vectors.
  virtual int64_t rank() const noexcept = 0;
  virtual int64_t num_directions() const noexcept = 0;
  /// @return Sampling units: num_directions, or num_directions num_groups
  /// for pairs.
  virtual int64_t num_units() const noexcept = 0;
  /// @return Groups of a basis vector: the set's (directions) or 1 (pairs).
  virtual int64_t basis_groups() const noexcept = 0;
  /// @return (r, basis_groups, num_space) the basis.
  virtual torch::Tensor basis() const = 0;
};

} // namespace ttnte::galerkin
