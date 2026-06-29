#pragma once

#include "ttnte/mesh/mesh_block_boundary.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/label.hpp"
#include <memory>
#include <optional>
#include <torch/extension.h>
#include <utility>

namespace ttnte::mesh {

/// @class MeshBlock
/// @brief Base class for all mesh blocks.
/// @tparam Derived The actual mesh block implementation (e.g. Patch)
template<typename Derived>
class MeshBlock : public std::enable_shared_from_this<Derived> {
public:
  // =================================================================
  // Public types
  /// Shared pointer to the Derived class.
  using Ptr = std::shared_ptr<Derived>;
  /// Shared pointer to the constant Derived class.
  using CPtr = std::shared_ptr<const Derived>;

  /// MeshBlock label type.
  using Label = utils::Label<MeshBlock>;
  /// Vector of Boundary type
  using Boundaries = c10::SmallVector<BoundaryInfo, 6>;

protected:
  // =================================================================
  // Protected data
  /// Local (to MPI rank) label or prescribed by the user
  Label label_;
  /// Block of data for mesh parameters
  torch::Tensor meshdata_;
  /// Vector of connected neighbors
  Boundaries boundaries_;
  /// The material or whatever that fills this region
  uint64_t fill_id_ = 0;
  /// Global ID
  int64_t gid_;

  /// Is this block immutable?
  bool is_finalized_ = false;

  // =================================================================
  // Protected constructors
  MeshBlock(std::optional<std::string> label = std::nullopt)
    : label_(label.has_value() ? Label::from_string(*label)
                               : Label::create_internal())
  {}
  MeshBlock(const Label& label) : label_(label) {}

  // =================================================================
  // Protected methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return std::string(Derived::class_name) + "::" + func_name;
  }

  void is_finalized_or_error(const std::string& func_name) const
  {
    if (!is_finalized_) {
      throw utils::runtime_error(derived(), error_context(func_name),
        "This class must be finalized before running this method");
    }
  }
  void is_not_finalized_or_error(const std::string& func_name) const
  {
    if (is_finalized_) {
      throw utils::runtime_error(derived(), error_context(func_name),
        "This class has already been finalized");
    }
  }

public:
  // =================================================================
  // Public construction methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new Derived(std::forward<Args>(args)...));
  }

  // =================================================================
  // Public methods
  /// @return Is this block immutable?
  bool is_finalized() const noexcept { return is_finalized_; }
  /// @brief Check the data and make this block immutable.
  void finalize()
  {
    auto& d = derived();
    d.finalize_impl();
    assert(meshdata_.defined());

    // Populate boundaries with UNKNOWN boundaries
    boundaries_.reserve(d.get_ndim() * 2);
    for (size_t dim = 0; dim < d.get_ndim(); dim++) {
      for (bool is_upper : {false, true}) {
        boundaries_.emplace_back(dim, is_upper);
      }
    }

    is_finalized_ = true;
  }

  /// @brief Add a connection to a boundary. A connection is the coupled face of
  /// another MeshBlock.
  /// @param dim The dimension for which we take upper or lower for the
  /// boundary.
  /// @param is_upper Whether to do the minimum or maximum of the dimension.
  /// @param ninfo The information of the neighboring MeshBlock and the boundary
  /// we're coupling to.
  void add_connection(size_t dim, bool is_upper, const NeighborInfo& ninfo)
  {
    assert(dim < get_ndim());
    // Connect this boundary to another block's boundary and set this as an
    // internal boundary
    auto& boundary = boundaries_[dim * 2 + static_cast<size_t>(is_upper)];
    boundary.add_connection(ninfo);
    boundary.set_type(physics::BoundaryType::INTERNAL);
  }

  /// @return A shortened string of this class for printing.
  std::string to_string() const { return derived().to_string_impl(); }

  /// @brief Send this block to another device or data type (in-place).
  /// @param options The tensor options.
  void to_(const torch::TensorOptions& options)
  {
    is_finalized_or_error("to_");
    meshdata_ = meshdata_.to(options);
  }

  // =================================================================
  // Public Getters / Setters
  /// @return Get a static pointer to the derived class.
  inline Derived& derived() { return *static_cast<Derived*>(this); }
  /// @return Get a static pointer to the constant derived class.
  const inline Derived& derived() const
  {
    return *static_cast<const Derived*>(this);
  }
  /// @return Get a shared pointer to the derived class.
  inline Ptr get_ptr() { return this->shared_from_this(); }
  /// @return Get the label of this class.
  const inline Label& get_label() const noexcept { return label_; }
  /// @return Get the device the mesh data is on.
  torch::Device get_device() const
  {
    is_finalized_or_error("get_device");
    return meshdata_.device();
  }
  /// @return Get the data type of the mesh data.
  torch::ScalarType get_dtype() const
  {
    is_finalized_or_error("get_dtype");
    return meshdata_.scalar_type();
  }
  /// @tparam FillType
  /// @return Get the label of the fill.
  template<typename FillType>
  utils::Label<FillType> get_fill() const noexcept
  {
    return utils::Label<FillType>(fill_id_);
  }
  /// @brief Get the bounding box of this block.
  /// @param epsilon The offset to apply to the bounding box.
  /// @return The minimum [0, :] and maximum points [1, :]
  torch::Tensor get_bbox(double epsilon = 0.0) const
  {
    return derived().get_bbox_impl(epsilon);
  }
  /// @brief Get the boundary block for this block.
  /// @param dim The dimension to take the upper or lower face.
  /// @param is_upper Whether to take the upper or lower face.
  /// @return A shared pointer to the new mesh block for the boundary
  Ptr get_boundary(size_t dim, bool is_upper)
  {
    // TODO: Consider how mesh data should change for this face
    return std::make_shared<Derived>(
      derived().get_boundary_impl(dim, is_upper));
  }
  /// @brief Get the number of mesh elements along a dimension.
  /// @param dim The dimension of interest.
  /// @return The number of elements.
  int64_t get_numel(size_t dim) const { return derived().get_numel_impl(dim); }
  /// @return The number of elements in the block.
  int64_t get_numel() const
  {
    const auto& d = derived();
    int64_t numel = 1;
    for (size_t dim = 0; dim < get_ndim(); dim++) {
      numel *= d.get_numel_impl(dim);
    }
    return numel;
  }
  /// @return Get the number of dimensions.
  int64_t get_ndim() const { return derived().get_ndim_impl(); }
  /// @return Get mesh coordinates
  torch::Tensor get_coords() const { return derived().get_coords_impl(); }
  /// @return The number of degrees of freedom (DOFs)
  int64_t get_num_dofs() const { return derived().get_num_dofs_impl(); }
  /// @return The global ID of the mesh block
  int64_t get_gid() const noexcept { return gid_; }

  /// @brief Get the boundary information at the top or bottom of a specific
  /// dimension.
  /// @param dim The dimension to take the upper or lower face.
  /// @param is_upper Whether to take the upper or lower face.
  /// @return The boundary information for that boundary.
  const BoundaryInfo& get_boundary_info(size_t dim, bool is_upper) const
  {
    is_finalized_or_error("get_interface");
    return boundaries_[dim * static_cast<size_t>(2) +
                       static_cast<size_t>(is_upper)];
  }
  /// @brief Get the boundary information at the top or bottom of a specific
  /// dimension (non const).
  /// @param dim The dimension to take the upper or lower face.
  /// @param is_upper Whether to take the upper or lower face.
  /// @return The boundary information for that boundary.
  BoundaryInfo& get_boundary_info(size_t dim, bool is_upper)
  {
    is_finalized_or_error("get_interface");
    return boundaries_[dim * static_cast<size_t>(2) +
                       static_cast<size_t>(is_upper)];
  }
  /// @return The information for all boundaries.
  const Boundaries& get_boundary_info() const noexcept { return boundaries_; }
  /// @return The information for all boundaries (non const).
  Boundaries& get_boundary_info() { return boundaries_; }
  /// @return The fill ID.
  const uint64_t& get_fill_id() const noexcept { return fill_id_; }

  /// @param label The new label.
  void inline set_label(const Label& label) { label_ = label; }
  /// @param label The new label.
  void inline set_label(const std::string& label)
  {
    label_ = Label::from_string(label);
  }
  /// @param id The new fill ID.
  void set_fill_id(const uint64_t& id) { fill_id_ = id; }
  /// @tparam FillType
  /// @param label Set the new fill using its label.
  template<typename FillType>
  void set_fill(const utils::Label<FillType>& label)
  {
    fill_id_ = label.to_int();
  }

  /// @brief Set the boundary condition type for a boundary.
  /// @param dim The dimension to take the upper or lower face.
  /// @param is_upper Whether to take the upper or lower face.
  /// @param type The new boundary condition.
  void set_boundary_type(
    size_t dim, bool is_upper, const physics::BoundaryType& type)
  {
    boundaries_[dim * static_cast<size_t>(2) + static_cast<size_t>(is_upper)]
      .set_type(type);
  }

  /// @param new_gid The new global ID for this MeshBlock.
  void set_gid(int64_t new_gid) { gid_ = new_gid; }
};

} // namespace ttnte::mesh
