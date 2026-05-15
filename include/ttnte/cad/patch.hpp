#pragma once

#include "ttnte/cad/bspline_basis.hpp"
#include "ttnte/mesh/mesh_block.hpp"
#include "ttnte/utils/label.hpp"
#include "ttnte/utils/mpi_helpers.hpp"
#include <optional>
#include <string>
#include <torch/extension.h>

namespace ttnte::cad {

// TODO: Methods to implement
// [ ] Inverse spline evaluation
// [ ] Basis function method (list of coordinates)
// [ ] Quadrature basis function method
// [ ] Gradient Basis function method (list of coordinates)
// [ ] Quadrature gradient basis function method
// [ ] Jacobian method (list of coordinates)
// [ ] Jacobian method on quadrature (out source to torchTT for TT)
// [ ] Normal evaluation (list of coordinates)
// [ ] Quadrature normal evaluate
// [ ] Initialize boundary method
// [ ] Boundary orientation method
// [ ] Load from hdf5 method
// [ ] Save to hdf5 method
// [ ] Knot insertion
// [ ] Order elevation
// [ ] A general refine method that uses knot insertion / order elevation
// [ ] Split patch method
// [ ] Functions related to solution fields (support both tensors and TTs)
// [ ] Functions related to MPI communication

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

  /// @brief Evaluate a tensor of parametric coordinates.
  /// @param local_coords Parametric coordinates shaped (n, m) where n
  /// is the number of points and m is the number of parametric dimensions.
  /// @return A tensor shaped (n, d) where d is the number of physical
  /// dimensions.
  torch::Tensor evaluate(const torch::Tensor& local_coords);
  /// @brief Evaluate a tensor product of parametric coordinates.
  /// @param local_coords A vector of 1-D tensors one for each parametric
  /// dimension.
  /// @return A tensor shaped (n1, ..., nk, d) where n1, ..., nk are the lengths
  /// of each input tensor and d is the number of physical dimensions.
  torch::Tensor evaluate(
    const c10::SmallVector<torch::Tensor, 3>& local_coords);

  // Compute all non-vanishing B-spline/NURBS basis functions.
  //
  // local_coords:
  //   Vector of 1D tensors, one per parametric dimension.
  //
  // Returns:
  //   Tensor of shape (n1, ..., nk, p1+1, ..., pk+1)
  torch::Tensor evaluate_basis(
    const c10::SmallVector<torch::Tensor, 3>& local_coords);

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

  // Get bounding box implementation
  torch::Tensor get_bbox_impl(double epsilon = 0.0) const;

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
  torch::Tensor get_coords_impl() const noexcept { return get_ctrlptsw(); }
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
