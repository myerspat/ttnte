#pragma once

#include "ttnte/sweep/patch_discretization.hpp"
#include <array>
#include <complex>
#include <memory>
#include <vector>

namespace ttnte::sweep {

/// @brief Exact local solver of patches whose operator is a Kronecker sum
/// (axis-aligned affine patches): with F_i the 1-D mass factors and
/// A_i^{+/-} the 1-D streaming plus outflow operators of a unit direction
/// along +/- axis i,
///   L(Omega, sigma) = (F_1 x ... x F_d) (sigma + sum_i |Omega_i| T_i^{sign}),
///   T_i^{sign} = F_i^{-1} A_i^{sign}.
/// Each T_i^{sign} has a complex Schur form U R U^*, computed once per patch
/// and shared by every direction and group, so a solve is d mode products,
/// a triangular tensor back substitution and d more mode products:
/// O(N sum_i n_i) operations against O(N b^2) for banded LU.
class KroneckerSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<KroneckerSolver>;
  using cplx = std::complex<double>;

  /// @brief Scratch of one worker.
  struct Workspace {
    std::vector<cplx> f;
    std::vector<cplx> g;
  };

private:
  // =================================================================
  // Private data
  int64_t ndim_ = 0;
  std::vector<int64_t> n_;
  /// Row-major strides of the flat dof index over n_.
  std::vector<int64_t> stride_;
  int64_t num_dofs_ = 0;
  /// Per dimension and sign (0: Omega_i >= 0, 1: Omega_i < 0), row-major
  /// (n_i, n_i): P = U^* F^{-1}, U and the upper triangular R.
  std::array<std::array<std::vector<cplx>, 2>, 3> P_;
  std::array<std::array<std::vector<cplx>, 2>, 3> U_;
  std::array<std::array<std::vector<cplx>, 2>, 3> R_;

  // =================================================================
  // Private constructors (see the static factories)
  KroneckerSolver() = default;

  /// @brief out = B applied along dimension `dim` of a C-order tensor.
  void mode_product(
    const cplx* B, int64_t dim, const cplx* in, cplx* out) const;

public:
  // =================================================================
  // Public methods
  /// @return The solver, or nullptr if the patch operator is not a
  /// Kronecker sum to relative accuracy `tol` (curved or skewed patches).
  /// Only unit directions are checked: L is linear in |Omega_i| for a fixed
  /// sign pattern on axis-aligned faces, and any other geometry fails the
  /// Kronecker check itself.
  static Ptr try_create(const PatchDiscretization& disc, double tol = 1e-10);

  /// @brief The solver of L = sigma (F_1 x ... x F_d) + sum_i A_i(Omega_i)
  /// x F_others from explicit 1-D factors.
  /// @param F Per dimension the (n_i, n_i) row-major mass factor.
  /// @param A Per dimension the (n_i, n_i) row-major operators of a unit
  /// direction along +axis i and -axis i.
  static Ptr from_factors(const std::vector<int64_t>& shape,
    const std::vector<std::vector<double>>& F,
    const std::vector<std::array<std::vector<double>, 2>>& A);

  /// @brief The Kronecker-sum operator of the patch's parametric twin: the
  /// same 1-D B-spline bases (weights ignored) on the parametric box, with
  /// the physical metric replaced by its patch averages. Solving it with
  /// direction metric * Omega and cross section measure * sigma
  /// approximates the curved patch's operator (a preconditioner).
  /// @param metric (ndim, ndim) average of |det J| J^{-1} over the patch.
  /// @param measure Average |det J| over the patch.
  static Ptr parametric_twin(
    const PatchDiscretization& disc, torch::Tensor& metric, double& measure);

  /// @brief Solve L(omega, sigma) x = b in place.
  /// @param omega (ndim) direction.
  /// @param sigma Total cross section.
  /// @param b (N) right-hand side, overwritten with x.
  /// @return true: a direct solve always succeeds (the same contract as
  /// IterativeSolver::solve(), whose false sends the caller to LU).
  bool solve(
    const double* omega, double sigma, double* b, Workspace& work) const;

  // =================================================================
  // Public getters
  int64_t num_dofs() const noexcept { return num_dofs_; }
  const std::vector<int64_t>& shape() const noexcept { return n_; }
};

} // namespace ttnte::sweep
