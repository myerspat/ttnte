#pragma once

#include "ttnte/sweep/patch_discretization.hpp"
#include <array>
#include <complex>
#include <memory>
#include <vector>

namespace ttnte::sweep {

/// Default largest condition number of the axial eigenvectors for
/// ExtrudedSolver's diagonal path (its error grows like cond * eps): about
/// 7e2 at 11 axial functions, 2e5 at 19 (p = 3), so finer axial meshes take
/// the Schur path.
inline constexpr double DEFAULT_MAX_EIGVEC_COND = 1e4;

/// @brief Exact local solver of extruded 3-D patches: a patch whose mapping
/// is x(xi) = (x_plane(the two other parametric axes), x_e(xi_s)), i.e.
/// parametric axis s maps to physical axis e alone and the cross-section
/// does not change along it (cad.extrude of a curved surface). Its Jacobian
/// is block diagonal, so the local operator is partially separable:
///   L(Omega, sigma) = L_p(Omega_p, sigma) (x) F_s + |Omega_e| M_p (x)
///   A_s^{sign},
/// with M_p the in-plane mass, L_p = sigma M_p + S_p(Omega_p) the in-plane
/// collision, streaming and side-face outflow (any curvature), and F_s,
/// A_s^{+/-} the 1-D mass and unit streaming plus outflow factors along s
/// (as in KroneckerSolver). With the complex Schur form
/// F_s^{-1} A_s^{sign} = U R U^*, shared by every direction and group, a
/// solve is one shifted in-plane band system per mode j of axis s, last mode
/// first:
///   (L_p + |Omega_e| R_jj M_p) w_j = (P b)_j - |Omega_e| M_p sum_{k>j} R_jk
///   w_k,
/// P = U^* F_s^{-1}, x = Re(U w): n_s complex band factorizations of the
/// in-plane order, against one real factorization of the whole patch for
/// banded LU, whose bandwidth is about n_s times larger. Omega_e = 0
/// decouples the modes (one real in-plane factorization).
///
/// The eigenvalues of the real, non-normal F_s^{-1} A_s come in conjugate
/// pairs (upwind advection; rarely real), so the solve diagonalizes it
/// instead when its eigenvectors are well conditioned
/// (try_create()'s max_eigvec_cond): the modes then decouple, b real makes the
/// solutions of a conjugate pair conjugate (one complex solve per pair,
/// x += 2 Re(v w)), and a real eigenvalue's mode is a real solve. About half
/// the complex work of the Schur path, which remains for larger n_s.
///
/// Every factor is read from the patch's own pair values (no new
/// quadrature), and try_create() checks the whole structure numerically.
class ExtrudedSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<ExtrudedSolver>;
  using cplx = std::complex<double>;

  /// @brief Scratch of one worker.
  struct Workspace {
    std::vector<double> values;
    std::vector<double> rhs;
    std::vector<double> layers;
    std::vector<double> clip;
    std::vector<double> real_band;
    std::vector<cplx> f;
    std::vector<cplx> t;
    std::vector<cplx> band;
    std::vector<int64_t> pivots;
  };

private:
  // =================================================================
  // Private types
  /// In-plane entries of one side face (a face whose normal has no
  /// component along physical axis e): the face points and, per entry, its
  /// point, in-plane pair and value at the reference layer of axis s.
  struct SideFace {
    int64_t num_points = 0;
    std::vector<double> normal;  // (Q, 3)
    std::vector<double> weights; // (Q)
    std::vector<int64_t> q;
    std::vector<int64_t> pair;
    std::vector<double> value;
  };

  // =================================================================
  // Private data
  /// Separable parametric axis and the physical axis it maps to.
  int64_t axis_ = 0;
  int64_t physical_ = 0;
  /// Modes along the axis, in-plane dofs, and the patch's dofs.
  int64_t ns_ = 0;
  int64_t m_ = 0;
  int64_t num_dofs_ = 0;
  /// Flat patch index of (mode j, in-plane band position a): (ns, m).
  std::vector<int64_t> flat_;
  /// In-plane pairs: band storage index, mass and the physical gradient
  /// components (3, P_p; the component along e unused).
  int64_t kl_ = 0;
  int64_t ku_ = 0;
  int64_t ldab_ = 0;
  std::vector<int64_t> pair_row_;
  std::vector<int64_t> pair_col_;
  std::vector<int64_t> band_index_;
  std::vector<double> mass_;
  std::vector<double> grad_;
  std::vector<SideFace> faces_;
  /// F_s^{-1} (real, for Omega_e = 0) and, per sign of Omega_e (0: >= 0),
  /// row-major (ns, ns) P = U^* F_s^{-1}, U and the upper triangular R.
  std::vector<double> Finv_;
  std::array<std::vector<cplx>, 2> P_;
  std::array<std::vector<cplx>, 2> U_;
  std::array<std::vector<cplx>, 2> R_;
  /// Per sign, the diagonal path (empty: Schur): one mode per real
  /// eigenvalue and per conjugate pair (its Im > 0 member) of
  /// F_s^{-1} A_s = V diag(lambda) V^{-1}; Q (k, ns) the modes' rows of
  /// V^{-1} F_s^{-1}, V (ns, k) their eigenvectors (real for a real mode).
  struct Modes {
    std::vector<cplx> lambda;
    std::vector<uint8_t> real;
    std::vector<cplx> Q;
    std::vector<cplx> V;
  };
  std::array<Modes, 2> modes_;

  /// @brief The diagonal path of T = F_s^{-1} A_s, or empty modes if T's
  /// eigenvectors are too ill-conditioned (or a conjugate pair is missing).
  static Modes diagonalize(
    const torch::Tensor& T, const torch::Tensor& Finv, double max_cond);

  // =================================================================
  // Private constructors (see try_create())
  ExtrudedSolver() = default;

  /// @brief The patch's structure with axis s mapped to physical axis e,
  /// or nullptr if it does not hold to relative accuracy `tol`.
  static Ptr try_axis(const PatchDiscretization& disc, int64_t s, int64_t e,
    double tol, double max_eigvec_cond);

  /// @brief In-plane pair values of sigma M_p + S_p(Omega_p) into `values`.
  void assemble_plane(
    const double* omega, double sigma, double* values, double* clip) const;

public:
  // =================================================================
  // Public methods
  /// @return The solver of an extruded 3-D patch, or nullptr if the patch
  /// is not extruded along any parametric axis (to relative accuracy `tol`
  /// in every pair value: the mass, the streaming of the axial unit
  /// directions, and the full operator of several oblique directions).
  /// Patches that are a full Kronecker sum (KroneckerSolver) pass too; the
  /// caller tries that solver first.
  /// @param max_eigvec_cond Largest condition number of the axial
  /// eigenvectors for the diagonal path (error about cond * eps; see
  /// DEFAULT_MAX_EIGVEC_COND); above it the Schur path. 0 forces Schur.
  static Ptr try_create(const PatchDiscretization& disc, double tol = 1e-10,
    double max_eigvec_cond = DEFAULT_MAX_EIGVEC_COND);

  /// @brief Solve L(omega, sigma) x = b in place.
  /// @param omega (3) direction.
  /// @param sigma Total cross section.
  /// @param b (N) right-hand side, overwritten with x.
  /// @return true (a direct solve), the contract of KroneckerSolver::solve().
  /// @throws std::runtime_error If an in-plane system is singular.
  bool solve(
    const double* omega, double sigma, double* b, Workspace& work) const;

  // =================================================================
  // Public getters
  int64_t num_dofs() const noexcept { return num_dofs_; }
  /// @return The separable parametric axis.
  int64_t axis() const noexcept { return axis_; }
  /// @return The physical axis it maps to.
  int64_t physical_axis() const noexcept { return physical_; }
  /// @return (kl, ku) of the in-plane band systems.
  std::pair<int64_t, int64_t> plane_bandwidth() const noexcept
  {
    return {kl_, ku_};
  }
};

} // namespace ttnte::sweep
