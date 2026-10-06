#pragma once

#include "ttnte/cad/patch.hpp"
#include <cstdint>
#include <memory>
#include <torch/extension.h>
#include <vector>

namespace ttnte::sweep {

/// @brief Quadrature data of one patch face, used for the upwind face
/// integrals of the sweep.
///
/// Face integrals follow the TT assembler exactly
/// (DGFirstOrderTransportBackend::assemble_boundary_geometry()): the outward
/// normal is the SCALED normal (tangent cross product, so |n| carries the
/// surface Jacobian) and `weights` is the parametric measure (knot span width
/// times normalized Gauss-Legendre weight), so
///   int_face f (Omega . n)_+ dS  ~=  sum_q f_q max(Omega . n_q, 0) w_q.
struct FaceGeometry {
  /// Parametric dimension the face is normal to.
  int64_t dim = 0;
  /// Upper (parameter = 1) or lower (parameter = 0) face.
  bool is_upper = false;
  /// Shape of the face's control-point grid (the volume grid with `dim`
  /// removed, C order). Empty for the end point of a 1-D patch.
  std::vector<int64_t> shape;
  /// (Nf) int64 volume (flat) index of each face degree of freedom.
  torch::Tensor dofs;
  /// (Qf, ndim) scaled outward normal at the face quadrature points.
  torch::Tensor normal;
  /// (Qf) parametric quadrature measure.
  torch::Tensor weights;
  /// Face mass integrand as a pair list over (quadrature point, test dof,
  /// trial dof): value R_a(q) R_b(q), dofs face-local. The quadrature weight
  /// enters through upwind_weights().
  torch::Tensor pair_q;
  torch::Tensor pair_a;
  torch::Tensor pair_b;
  torch::Tensor pair_val;
  /// (P_f) int64 position of each face pair in the patch's coalesced volume
  /// pair list (face dofs only couple to dofs they share volume support with).
  torch::Tensor pair_volume_index;
  /// Physical measure of the face, sum_q |n_q| w_q.
  double area = 0.0;

  /// @return Number of face degrees of freedom.
  int64_t num_dofs() const { return dofs.size(0); }

  /// @brief Upwind face integrals per direction.
  /// @param omega (D, ndim) directions.
  /// @param outflow True for (Omega . n)_+, false for (Omega . n)_-.
  /// @return (D, Qf) clipped and weighted normal components.
  torch::Tensor upwind_weights(const torch::Tensor& omega, bool outflow) const;
};

/// @brief Raw pointers into one face's FaceGeometry tensors (valid for the
/// lifetime of the owning PatchDiscretization).
struct FaceView {
  int64_t num_dofs = 0;
  int64_t num_points = 0;
  int64_t num_pairs = 0;
  const int64_t* dofs = nullptr;
  const double* normal = nullptr;
  const double* weights = nullptr;
  const int64_t* pair_q = nullptr;
  const int64_t* pair_a = nullptr;
  const int64_t* pair_b = nullptr;
  const double* pair_val = nullptr;
  const int64_t* pair_volume_index = nullptr;
};

/// @brief Raw pointers into a PatchDiscretization's tensors, built once so
/// inner kernel loops never touch the tensor API.
struct KernelView {
  int64_t ndim = 0;
  int64_t num_dofs = 0;
  int64_t num_pairs = 0;
  int64_t kl = 0;
  int64_t ku = 0;
  int64_t ldab = 0;
  int64_t max_face_points = 0;
  const int64_t* pair_row = nullptr;
  const int64_t* pair_col = nullptr;
  const double* mass = nullptr;
  /// (ndim, P)
  const double* grad = nullptr;
  const int64_t* band_index = nullptr;
  const int64_t* dense_index = nullptr;
  std::vector<FaceView> faces;
};

/// @brief Discrete operators of one IGA patch for the discrete-ordinates sweep.
///
/// The patch's local system for direction d and group g is
///   L_{d,g} = sigma_{t,g} M - sum_i Omega_{d,i} G_i + sum_f B_f^+(d),
/// with mass M_{vu} = (R_v, R_u), streaming G_{i,vu} = (d_i R_v, R_u) and
/// outflow face terms B_f^+(d)_{vu} = <(Omega . n)_+ R_v, R_u>_f, the exact
/// weak form assembled by the TT solver (rows are test functions). Volume
/// integrals are assembled span by span from the non-vanishing NURBS basis
/// functions on the same Gauss-Legendre points (p + 1 per knot span) the TT
/// assembler uses, and stored as a coalesced pair list so the operator can be
/// written directly into banded or dense storage. The same class serves patch
/// (PDG) and single-element (FDG) discretizations.
///
/// Every tensor member is contiguous CPU float64/int64, so raw-array kernels
/// (see sweep_kernel.hpp) may read them through data_ptr().
class PatchDiscretization {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<PatchDiscretization>;

protected:
  // =================================================================
  // Protected data
  /// Number of parametric (= physical) dimensions.
  int64_t ndim_ = 0;
  /// Control-point grid shape.
  std::vector<int64_t> shape_;
  /// Number of degrees of freedom.
  int64_t num_dofs_ = 0;
  /// (P) coalesced volume pair rows (test dof) and columns (trial dof).
  torch::Tensor pair_row_;
  torch::Tensor pair_col_;
  /// (P) mass values.
  torch::Tensor mass_val_;
  /// (ndim, P) streaming values (d_i R_row, R_col).
  torch::Tensor grad_val_;
  /// Faces, indexed 2 * dim + is_upper.
  std::vector<FaceGeometry> faces_;
  /// Sub- and super-diagonal bandwidth of the operator pattern.
  int64_t kl_ = 0;
  int64_t ku_ = 0;
  /// (P) flat index of each pair in (n, 2kl+ku+1) band storage.
  torch::Tensor band_index_;
  /// (P) flat index of each pair in row-major (n, n) dense storage.
  torch::Tensor dense_index_;
  /// (N) integral of each basis function (load vector of a unit source).
  torch::Tensor basis_integral_;
  /// Raw pointers into the tensors above.
  KernelView view_;

  // =================================================================
  // Protected constructors
  explicit PatchDiscretization(const cad::Patch& patch);

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new PatchDiscretization(std::forward<Args>(args)...));
  }

  /// @brief Streaming plus outflow part of the operator for a batch of
  /// directions (no collision term), as pair values.
  /// @param omega (D, ndim) directions.
  /// @return (D, P) pair values of -sum_i Omega_i G_i + sum_f B_f^+.
  torch::Tensor streaming_values(const torch::Tensor& omega) const;

  /// @brief Write pair values into band storage ready for factorization.
  /// @param values (B, P) pair values.
  /// @return (B, N, 2kl+ku+1) band storage.
  torch::Tensor to_band(const torch::Tensor& values) const;

  /// @brief Write pair values into dense matrices.
  /// @param values (B, P) pair values.
  /// @return (B, N, N) dense matrices (rows test, columns trial).
  torch::Tensor to_dense(const torch::Tensor& values) const;

  /// @brief Apply the mass matrix to a batch of coefficient vectors.
  /// @param x (B, N) coefficients.
  /// @return (B, N) M x.
  torch::Tensor apply_mass(const torch::Tensor& x) const;

  /// @brief Full local operator for one direction and total cross section,
  /// dense (for testing and small problems).
  /// @param omega (ndim) direction.
  /// @param sigma_t Total cross section.
  /// @return (N, N) L = sigma_t M - Omega . G + sum_f B_f^+.
  torch::Tensor local_operator(
    const torch::Tensor& omega, double sigma_t) const;

  // =================================================================
  // Public getters
  /// @return Raw pointers into this discretization for kernels.
  const KernelView& view() const noexcept { return view_; }
  int64_t ndim() const noexcept { return ndim_; }
  const std::vector<int64_t>& shape() const noexcept { return shape_; }
  int64_t num_dofs() const noexcept { return num_dofs_; }
  int64_t num_pairs() const { return pair_row_.size(0); }
  int64_t kl() const noexcept { return kl_; }
  int64_t ku() const noexcept { return ku_; }
  /// @return Leading dimension of the band storage, 2 kl + ku + 1.
  int64_t ldab() const noexcept { return 2 * kl_ + ku_ + 1; }
  const torch::Tensor& band_index() const noexcept { return band_index_; }
  const torch::Tensor& dense_index() const noexcept { return dense_index_; }
  const torch::Tensor& pair_row() const noexcept { return pair_row_; }
  const torch::Tensor& pair_col() const noexcept { return pair_col_; }
  const torch::Tensor& mass_values() const noexcept { return mass_val_; }
  const torch::Tensor& grad_values() const noexcept { return grad_val_; }
  const torch::Tensor& basis_integral() const noexcept
  {
    return basis_integral_;
  }
  const FaceGeometry& face(size_t dim, bool is_upper) const
  {
    return faces_.at(2 * dim + static_cast<size_t>(is_upper));
  }
  const std::vector<FaceGeometry>& faces() const noexcept { return faces_; }
};

} // namespace ttnte::sweep
