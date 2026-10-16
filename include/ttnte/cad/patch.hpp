#pragma once

#include "ttnte/cad/bspline_basis.hpp"
#include "ttnte/mesh/mesh_block.hpp"
#include "ttnte/utils/label.hpp"
#include <optional>
#include <string>
#include <torch/extension.h>

namespace ttnte::cad {

/// Default number of points per parametric axis for inverse_map()'s
/// coarse-grid closest-point seeding, when no explicit initial_guess is
/// given. Chosen empirically as the smallest tested power-of-round-number
/// resolution that reliably seeds Newton-Raphson into the correct basin near
/// coordinate singularities (e.g. a sphere octant patch's pole, where an
/// entire parametric edge collapses to one physical point) and multi-patch
/// corners where several patches meet at a single point (e.g. two 45-degree
/// arc segments meeting at a shared vertex) -- a lower resolution (e.g. the
/// legacy default of 10) can seed Newton into a nearby-but-wrong fixed
/// point near such features, causing spurious non-convergence even though
/// the target point is genuinely on the patch.
inline constexpr int64_t DEFAULT_INVERSE_MAP_SEED_RESOLUTION = 60;

/// @brief Result of Patch::inverse_map(): per-point parametric coordinates
/// found via Newton-Raphson, along with convergence diagnostics.
struct InverseMapResult {
  /// Best-effort parametric coordinates, shape (n, ndim).
  torch::Tensor coords;
  /// Final physical-space residual norm, shape (n,).
  torch::Tensor residual;
  /// Whether each point converged within tol/max_iter, shape (n,), bool.
  torch::Tensor converged;
};

class Patch : public mesh::MeshBlock<Patch> {
  friend class mesh::MeshBlock<Patch>;

public:
  // =================================================================
  // Public types
  using Label = utils::Label<Patch>;
  using Basis = c10::SmallVector<BSplineBasis, 3>;
  using Base = mesh::MeshBlock<Patch>;

  static constexpr const char* class_name = "ttnte::cad::Patch";

private:
  // =================================================================
  // Private data
  Basis basis_;
  bool is_rational_ = false;

  // B-Spline: stores (N_1, ..., N_D, dim)
  // NURBS: stores (N_1, ..., N_D, dim + 1) where the last dim is (wx, wy, wz)
  torch::Tensor ctrlptsw_;

  /// Orientation for this patch. If the patch is in a left-handed coordinate
  /// system then we scale by this to get it back to the right-handed coordinate
  /// system.
  double orientation_ = 1.0;

  // =================================================================
  // Private constructors
  Patch(std::optional<std::string> label = std::nullopt);

  Patch(const torch::Tensor& ctrlpts, const Basis& basis,
    bool is_rational = false, std::optional<std::string> label = std::nullopt);

  Patch(const torch::Tensor& ctrlpts, const torch::Tensor& weights,
    const Basis& basis, std::optional<std::string> label = std::nullopt);

  // =================================================================
  // Private methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return std::string("ttnte::cad::Patch::") + func_name;
  }

public:
  // =================================================================
  // Public methods
  /// @return Is this a NURBS (true) or a B-Spline (false)?
  bool is_rational() const noexcept { return is_rational_; }
  /// @brief This implements the finalize method called by MeshBlock<Patch>. All
  /// the geometric data (control points and basis) are checked, the single mesh
  /// data tensor is populated, and Patch is marked immutable.
  void finalize_impl();

  /// @brief Evaluate a tensor product of parametric coordinates.
  /// @param local_coords A vector of 1-D tensors one for each parametric
  /// dimension.
  /// @return A tensor shaped (n1, ..., nk, d) where n1, ..., nk are the lengths
  /// of each input tensor and d is the number of physical dimensions.
  torch::Tensor evaluate(
    const c10::SmallVector<torch::Tensor, 3>& local_coords) const;

  /// @brief Evaluate a tensor of independent parametric coordinates, including
  /// partial derivatives up to `derivative_order`. Mirrors the "no cross
  /// terms" convention of evaluate_all_basis()/evaluate_all_jacobian(): the
  /// derivative axis holds the value, then the `derivative_order`-th partial
  /// derivative in each parametric dimension in turn (holding the other
  /// dimensions at derivative order 0) -- not full mixed partials.
  /// @param local_coords Parametric coordinates shaped (n, m).
  /// @param derivative_order The order of derivative to compute to. If 0,
  /// this returns the same values as evaluate(local_coords) (but via a
  /// separate code path).
  /// @return A tensor shaped (n, d) if `derivative_order == 0`, else (n, d,
  /// 1 + ndim * derivative_order).
  torch::Tensor evaluate(
    const torch::Tensor& local_coords, int64_t derivative_order = 0) const;

  /// @brief Compute the Jacobian at a tensor of independent parametric
  /// coordinates. This is the evaluate(Tensor)-batch analogue of
  /// evaluate_all_jacobian(), which only supports tensor-product grids.
  /// @param local_coords Parametric coordinates shaped (n, m).
  /// @return A tensor of shape (n, n_physical, n_parametric).
  torch::Tensor evaluate_jacobian(const torch::Tensor& local_coords) const;

  /// @brief Invert the parametric-to-physical map for a batch of target
  /// physical points via batched (least-squares, Levenberg-Marquardt damped)
  /// Newton-Raphson, using evaluate()/evaluate_jacobian(). This considers
  /// only this one patch -- candidate-patch selection across multiple
  /// patches (e.g. a bounding-box pre-filter, or reconciling multiple
  /// patches' results) is the caller's responsibility.
  /// @param physical_coords Target physical points, shape (n, n_physical).
  /// @param max_iter Maximum number of Newton-Raphson iterations.
  /// @param tol Convergence tolerance on the physical-space residual norm.
  /// @param initial_guess Optional starting parametric coordinates, shape
  /// (n, ndim). If not given, a coarse-grid closest-point search seeds each
  /// point independently.
  /// @param seed_resolution Number of points per parametric axis for the
  /// coarse-grid closest-point search that seeds Newton-Raphson when
  /// `initial_guess` is not given (ignored otherwise). Raise this if points
  /// near a coordinate singularity (e.g. a collapsed patch edge) or a
  /// multi-patch corner spuriously fail to converge.
  /// @return Per-point best-effort coordinates, residuals, and convergence
  /// flags.
  InverseMapResult inverse_map(const torch::Tensor& physical_coords,
    int64_t max_iter = 10, double tol = 1e-8,
    std::optional<torch::Tensor> initial_guess = std::nullopt,
    int64_t seed_resolution = DEFAULT_INVERSE_MAP_SEED_RESOLUTION) const;

  /// @brief Evaluate a DOF-coefficient field defined on this patch's own
  /// basis (e.g. a solved scalar/angular-flux field's dense array) at a
  /// batch of independent parametric points. Directly contracts `field`
  /// against this patch's own basis functions (the same per-dimension
  /// contraction evaluate() uses internally), NURBS-dividing by the weight
  /// function if this patch is rational -- no synthetic patch/finalize()
  /// is involved, since finalize()'s Jacobian/dimensionality validity
  /// checks are only meaningful for real geometry, not arbitrary field
  /// data (e.g. a field patch with a single channel but ndim > 1 would
  /// otherwise spuriously fail the "parametric dim <= physical dim" check).
  /// @param field Dense field array, shape matching this patch's own
  /// control-point grid plus a trailing channel axis:
  /// (*ctrlpts_grid_shape, num_channels).
  /// @param points Parametric coordinates, shape (n, ndim).
  /// @return Field values at `points`, shape (n, num_channels).
  torch::Tensor evaluate_field(
    const torch::Tensor& field, const torch::Tensor& points) const;

  /// @brief Compute all non-vanishing B-spline/NURBS basis functions.
  /// @param local_coords The tensor product parametric coordinates.
  /// @param derivative_order The order of derivative to compute to.
  /// @return A tensor of shape (n1, ..., nk, p1 + 1, ...., pk + 1,
  /// derivative_order + 1). If `derivative_order == 1` then the last dimension
  /// is squeezed.
  torch::Tensor evaluate_basis(
    const c10::SmallVector<torch::Tensor, 3>& local_coords,
    const int64_t& derivative_order = 0) const;

  /// @brief Compute all B-spline/NURBS basis functions.
  /// @param local_coords The tensor product parametric coordinates.
  /// @param derivative_order The order of derivative to compute to.
  /// @param eval_cross_terms Whether to evaluate cross derivatives.
  /// @return If `eval_cross_terms == true` then a tensor of shape (n1, ..., nk,
  /// derivative_order + 1, ..., derivative_order + 1, c1, ..., ck) where ni is
  /// the number of parametric points and ci is the number of control points all
  /// along parametric dimension i. Note that there will be as many dimensions
  /// for derivatives as there are parametric dimensions. If `eval_cross_terms
  /// == false` when we compute no cross terms and the dimensions for the
  /// derivatives are collapsed into one where the first index are the zeroth
  /// derivative, the next `get_ndim()` are the first, and so on.
  torch::Tensor evaluate_all_basis(
    const c10::SmallVector<torch::Tensor, 3>& local_coords,
    int64_t derivative_order = 0, bool eval_cross_terms = false) const;

  /// @brief Compute the Jacobian at tensor product points.
  /// @param local_coords The tensor product parametric coordinates.
  /// @return A tensor of shape (n1, ..., nk, n_physical, n_parametric, p1 + 1).
  torch::Tensor evaluate_all_jacobian(
    const c10::SmallVector<torch::Tensor, 3>& local_coords) const;

  // Insert new knots following generalized version of Algorithim 5.5
  // from the NURBS book. Output new knot vector and ctrl_pts
  //
  // new_knots:
  //   Vector of 1D tensors, one per parametric dimension.
  //   Amount of times to insert knots.
  //
  // Action:
  //   Edits knotvector_ and ctrlptsw_.
  Patch knot_insert(const c10::SmallVector<torch::Tensor, 3>& new_knots,
    const int64_t& reps = 1);

  // Insert new knots following generalized version of Algorithim 5.5
  // from the NURBS book. Replace old knot vectors and ctrl pts with output
  //
  // new_knots:
  //   Vector of 1D tensors, one per parametric dimension.
  //   Amount of times to insert knots.
  //
  // Action:
  //   Edits knotvector_ and ctrlptsw_.
  void knot_insert_(const c10::SmallVector<torch::Tensor, 3>& new_knots,
    const int64_t& reps = 1);

  /// @brief Split the patch into single-element (Bezier) patches, one per
  /// knot span, by raising every interior knot to multiplicity p + 1. The
  /// geometry is unchanged; each element patch carries a [0, 1] Bernstein
  /// basis of the same degree, and inherits this patch's fill and fixed
  /// source. Used to build fully discontinuous (FDG) meshes from patch
  /// discontinuous (PDG) ones.
  /// @return The element patches in C order over the knot spans.
  /// @throws ttnte::utils::runtime_error If a projected (basis-dependent)
  /// fixed source is attached, since it cannot be carried over.
  std::vector<Ptr> split_elements() const;

  // // MPI communication
  // template<typename T>
  // void inline pack(
  //   std::vector<int64_t>& meta_buffer, std::vector<T>& payload_buffer) const
  // {
  //   // Fill the meta data buffer first
  //   meta_buffer.push_back(label_.to_int()); // Patch label
  //   meta_buffer.push_back(is_valid_);       // Is valid Boolean
  //   meta_buffer.push_back(is_rational_);    // Is rational Boolean
  //   meta_buffer.push_back(basis_.size());   // Number of parametric
  //   dimensions
  //
  //   // Iterate through the basis
  //   for (const auto& b : basis_) {
  //     b.pack(meta_buffer, payload_buffer);
  //   }
  //
  //   // Add the control point information
  //   meta_buffer.push_back(ctrlptsw_.numel()); // Number of control points
  //   auto ctrlptsw_c = ctrlptsw_.flatten().contiguous();
  //   payload_buffer.insert(payload_buffer.end(), ctrlptsw_c.data_ptr<T>(),
  //     ctrlptsw_c.data_ptr<T>() + ctrlptsw_c.numel());
  // }

  // void pack(std::vector<int64_t>& meta_buffer,
  //   std::vector<torch::Tensor>& payload_buffer) const;

  // template<typename T>
  // static inline Patch unpack(const int64_t* meta_buffer,
  //   const T* payload_buffer, int& meta_idx, int& payload_idx)
  // {
  //   // Patch metadata
  //   Label label = Label(static_cast<uint64_t>(meta_buffer[meta_idx++]));
  //   bool is_valid = static_cast<bool>(meta_buffer[meta_idx++]);
  //   bool is_rational = static_cast<bool>(meta_buffer[meta_idx++]);
  //   int64_t ndim = meta_buffer[meta_idx++];
  //
  //   // Build the basis
  //   Basis basis;
  //   basis.reserve(ndim);
  //   c10::SmallVector<int64_t, 4> shape;
  //   shape.reserve(ndim + 1);
  //
  //   for (size_t d = 0; d < ndim; d++) {
  //     basis.push_back(BSplineBasis::unpack(
  //       meta_buffer, payload_buffer, meta_idx, payload_idx));
  //     shape.push_back(basis.back().get_size());
  //   }
  //   shape.push_back(static_cast<int64_t>(-1));
  //
  //   // Control points
  //   torch::Tensor ctrlptsw =
  //     utils::unpack_tensor(&payload_buffer[payload_idx], std::move(shape),
  //       torch::TensorOptions()
  //         .device(torch::kCPU)
  //         .dtype(torch::CppTypeToScalarType<T>::value));
  //   payload_idx += meta_buffer[meta_idx++];
  //
  //   return Patch(std::move(ctrlptsw), std::move(basis),
  //   std::move(is_rational),
  //     std::move(is_valid), std::move(label));
  // }

  /// @return Class info in a string for printing.
  std::string to_string_impl() const;

  /// @brief Send the patch to a device or a different data type (in-place).
  /// @param options The tensor options.
  void to_(const torch::TensorOptions& options);

  /// @brief Send the patch to a device or a different data type (in-place).
  /// @param options The tensor options.
  Ptr to(const torch::TensorOptions& options) const;

  // =================================================================
  // Public Getters / Setters
  /// @return The control points (weighted for NUBRS) in the shape of (N1, ...,
  /// ND, phys_dims) where N1, ..., ND is the number of basis functions along
  /// each dimension.
  const torch::Tensor& get_ctrlptsw() const;
  /// @return The basis of the B-Spline or NURBS.
  const inline Basis& get_basis() const noexcept { return basis_; }
  /// @return The number of parametric dimensions.
  inline int64_t get_ndim_impl() const noexcept { return basis_.size(); }
  /// @return The control points (unweighted for NUBRS).
  torch::Tensor get_ctrlpts() const;
  /// @return The weights. This is a tensor of ones for B-Splines.
  torch::Tensor get_weights() const;
  /// @brief Get the polynomial order along a parametric dimension.
  /// @param dim The parametric dimension.
  /// @return The polynomial order.
  int64_t get_order(size_t dim) const;
  /// @return The order of the B-Spline or NURBS along each parametric
  /// dimension.
  std::vector<int64_t> get_orders() const;
  /// @brief Get the polynomial degree along a parametric dimension.
  /// @param dim The parametric dimension.
  /// @return The polynomial degree.
  int64_t get_degree(size_t dim) const;
  /// @return The polynomial degree of the B-Spline or NURBS along each
  /// parametric dimension.
  std::vector<int64_t> get_degrees() const;
  /// @brief Get the number of control points along a parametric dimension.
  /// @param dim The parametric dimension.
  /// @return The number of control points.
  int64_t get_ctrlpts_size(size_t dim) const;
  /// @return The number of control points along each parametric dimension.
  std::vector<int64_t> get_ctrlpts_sizes() const;

  /// @return Get bounding box. (Implementation)
  torch::Tensor get_bbox_impl(double epsilon = 0.0) const;

  /// @return Get the orientation of the patch (left- versus right-handed
  /// coordinate system)
  double get_orientation() const noexcept { return orientation_; }

  /// @brief Get the spline that defining the upper or lower boundary along a
  /// parametric dimension.
  /// @param dim The parametric dimension.
  /// @param is_upper If true we take the parametric dimension at 1 else we use
  /// 0.
  /// @return The boundary patch.
  Patch get_boundary_impl(size_t dim, bool is_upper);

  /// @brief Implementation to get the number of elements along a dimension.
  /// @param dim The dimension of interest.
  /// @return The number of elements.
  int64_t get_numel_impl(size_t dim) const;

  /// @return Get the coordinates for this mesh (control points)
  torch::Tensor get_coords_impl() const noexcept { return get_ctrlpts(); }
  /// @return Get the number of degrees of freedom implementation
  int64_t get_num_dofs_impl() const;

  /// @param ctrlptsw The new weighted control points.
  void set_ctrlptsw(const torch::Tensor& ctrlptsw);
  /// @param basis The new basis.
  void set_basis(const Basis& basis);
  /// @param ctrlpts The new unweighted control points.
  void set_ctrlpts(torch::Tensor ctrlpts);
  /// @param weights The new weights for the NURBS.
  void set_weights(torch::Tensor weights);
};

std::ostream& operator<<(std::ostream& os, const Patch& p);

} // namespace ttnte::cad
