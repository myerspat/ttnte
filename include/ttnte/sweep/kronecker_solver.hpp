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
  int64_t num_dofs_ = 0;
  /// Per dimension and sign (0: Omega_i >= 0, 1: Omega_i < 0), row-major
  /// (n_i, n_i): P = U^* F^{-1}, U and the upper triangular R.
  std::array<std::array<std::vector<cplx>, 2>, 3> P_;
  std::array<std::array<std::vector<cplx>, 2>, 3> U_;
  std::array<std::array<std::vector<cplx>, 2>, 3> R_;

  // =================================================================
  // Protected constructors
  KroneckerSolver() = default;

  /// @brief out = B applied along dimension `dim` of a C-order tensor.
  void mode_product(
    const cplx* B, int64_t dim, const cplx* in, cplx* out) const;

public:
  // =================================================================
  // Public methods
  /// @return The solver, or nullptr if the patch operator is not a
  /// Kronecker sum to relative accuracy `tol` (curved or skewed patches).
  static Ptr try_create(const PatchDiscretization& disc, double tol = 1e-10);

  /// @brief Solve L(omega, sigma) x = b in place.
  /// @param omega (ndim) direction.
  /// @param sigma Total cross section.
  /// @param b (N) right-hand side, overwritten with x.
  void solve(
    const double* omega, double sigma, double* b, Workspace& work) const;

  // =================================================================
  // Public getters
  int64_t num_dofs() const noexcept { return num_dofs_; }
  const std::vector<int64_t>& shape() const noexcept { return n_; }
};

} // namespace ttnte::sweep
