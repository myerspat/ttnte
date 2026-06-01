#pragma once

#include "ttnte/cad/patch.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/linalg/tt_operator.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include "ttnte/physics/boundary_types.hpp"
#include "ttnte/xs/server.hpp"
#include <c10/util/SmallVector.h>

namespace ttnte::physics::backends {

// =================================================================
// Return types

/// @brief Type templating for the different formats to return.
template<FormatType Fmt, int64_t NumDim>
struct Return;

/// @brief Return types for the dense format.
template<int64_t NumDim>
struct Return<FormatType::DENSE, NumDim> {
  using Type = torch::Tensor;
  using VectorType = torch::Tensor;
  using MatrixType = torch::Tensor;
  using OperatorType = linalg::TTOperator::Ptr;
};

/// @brief Return types for the tensor train format.
template<int64_t NumDim>
struct Return<FormatType::TENSOR_TRAIN, NumDim> {
  using Type = linalg::TTEngine;
  using VectorType = c10::SmallVector<linalg::TTEngine, NumDim>;
  using MatrixType = std::array<std::array<linalg::TTEngine, NumDim>, 3>;
  using OperatorType = linalg::TTOperator::Ptr;
};

// =================================================================
// Caches for different formats

/// @brief Templated class for caching components of operators to avoid repeated
/// evaluation.
template<typename BlockType, typename ConfigType, FormatType Fmt,
  int64_t NumDim>
class BackendCache {
public:
  // =================================================================
  // Public constructors
  BackendCache(
    const typename BlockType::Ptr& block, const ConfigType& config) {};

  // =================================================================
  // Public methods
  /// @brief Send the cache to another device or a different data type
  /// (in-place).
  /// @param options The tensor options applied to all tensors.
  void to_(const torch::TensorOptions& options) {}
};

/// @brief Specific implementation for B-spline and NURBS patches.
template<FormatType Fmt, int64_t NumDim>
class BackendCache<cad::Patch, DGAssemblerConfig, Fmt, NumDim> {
public:
  // =================================================================
  // Public types
  using TensorType = typename Return<Fmt, NumDim>::Type;
  using VectorType = typename Return<Fmt, NumDim>::VectorType;
  using MatrixType = typename Return<Fmt, NumDim>::MatrixType;

private:
  // =================================================================
  // Private data
  /// Weighted control points decomposed into individual.
  typename Return<Fmt, 3>::VectorType ctrlptrs_;
  /// The weights for a NURBS.
  mutable std::optional<TensorType> weights_;

  /// Lazy cache for the ordinates.
  mutable std::optional<VectorType> ordinates_;

  /// Lazy cache for the basis functions.
  mutable std::optional<TensorType> basis_;
  /// Lazy cache for the basis function derivatives.
  mutable std::optional<VectorType> ders_;

  /// Lazy cache for the Jacobian matrix.
  mutable std::optional<MatrixType> jacobian_;
  /// Lazy cache for the integral mapping (Jacobian determinant in most cases).
  mutable std::optional<TensorType> mapping_;
  /// Lazy cache for the Jacobian inverse (for mapping the gradient operator).
  mutable std::optional<MatrixType> jacobian_inverse_;

  /// Lazy cache for the weighted basis outer product.
  mutable std::optional<TensorType> mapped_basis_;

public:
  // =================================================================
  // Public constructors
  BackendCache(const cad::Patch::Ptr& block, const DGAssemblerConfig& config)
  {
    if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
      assert(block.get_ndim() == NumDim);

      // Get the control points from the patch
      const auto& ctrlptsw = block->get_ctrlptsw();

      // Config settings
      double eps = config.rounding.eps;
      int max_rank = config.rounding.max_rank;

      // Decompose the weighted control points into the TT format
      if (!block->is_rational()) {
        int64_t space_dim = ctrlptsw.size(-1);
        ctrlptrs_.reserve(space_dim);

        // B-spline case (just the control points)
        for (size_t i = 0; i < space_dim; i++) {
          ctrlptrs_.push_back(linalg::TTEngine::from_dense(
            ctrlptsw.select(-1, i), eps, max_rank));
          ctrlptrs_.back().transpose_();
        }

      } else {
        int64_t space_dim = ctrlptsw.size(-1) - 1;
        ctrlptrs_.reserve(space_dim);

        // NURBS case (unweight control points and put the weights at the end)
        torch::Tensor weights = ctrlptsw.select(-1, -1);
        for (size_t i = 0; i < space_dim; i++) {
          ctrlptrs_.push_back(linalg::TTEngine::from_dense(
            ctrlptsw.select(-1, i) / weights, eps, max_rank));
          ctrlptrs_.back().transpose_();
        }

        weights_ = linalg::TTEngine::from_dense(weights, eps, max_rank);
        weights_->transpose_();
      }
    }
  }

  // =================================================================
  // Public methods

  /// @brief Send the cache to another device or data type.
  /// @param options The tensor options.
  void to_(const torch::TensorOptions& options)
  {
    // Helper lambdas
    auto move_tensor = [&](auto& t) {
      if constexpr (Fmt == FormatType::DENSE) {
        t = t.to(options);
      } else if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
        t.to_(options);
      }
    };
    auto move_vector = [&](auto& ts) {
      if constexpr (Fmt == FormatType::DENSE) {
        move_tensor(ts);
      } else {
        for (size_t i = 0; i < ts.size(); ++i) {
          move_tensor(ts[i]);
        }
      }
    };
    auto move_matrix = [&](auto& ts) {
      if constexpr (Fmt == FormatType::DENSE) {
        move_tensor(ts);
      } else {
        for (auto& row : ts) {
          for (auto& elem : row) {
            move_tensor(elem);
          }
        }
      }
    };

    move_vector(ctrlptrs_);
    if (weights_)
      move_tensor(*weights_);
    if (basis_)
      move_tensor(*basis_);
    if (ders_)
      move_vector(*ders_);
    if (jacobian_)
      move_matrix(*jacobian_);
    if (mapping_)
      move_tensor(*mapping_);
    if (jacobian_inverse_)
      move_matrix(*jacobian_inverse_);
    if (mapped_basis_)
      move_tensor(*mapped_basis_);
  }

  // =================================================================
  // Public getters / setters

  /// @return The decomposed control points and weights.
  const auto& get_ctrlpts() const { return ctrlptrs_; }

  /// @return Whether the cache has the NURBS weights.
  bool has_weights() const { return weights_.has_value(); }
  /// @return The NURBS weights.
  const TensorType& get_weights() const { return *weights_; }

  /// @return Whether the cache has the ordinates.
  bool has_ordinates() const { return ordinates_.has_value(); }
  /// @return The ordinates.
  const VectorType& get_ordinates() const { return *ordinates_; }
  /// @param ordinates Set the ordinates.
  void set_ordinates(const VectorType& ordinates) { ordinates_ = ordinates; }

  /// @return Whether the cache has the basis.
  bool has_basis() const { return basis_.has_value(); }
  /// @return The basis.
  const TensorType& get_basis() const { return *basis_; }
  /// @param basis Set the basis.
  void set_basis(const TensorType& basis) { basis_ = basis; }

  /// @return Whether the cache has the basis function derivatives.
  bool has_ders() const { return ders_.has_value(); }
  /// @return The basis function derivatives.
  const VectorType& get_ders() const { return *ders_; }
  /// @param ders Set the basis function derivatives.
  void set_ders(const VectorType& ders) { ders_ = ders; }

  /// @return Whether the cache has the Jacobian mapping for the B-spline/NURBS
  /// patch.
  bool has_jacobian() const { return jacobian_.has_value(); }
  /// @return The Jacobian mapping who's dimensions are (physical dimensions,
  /// parametric dimensions)
  const MatrixType& get_jacobian() const { return *jacobian_; }
  /// @param jacobian Set the Jacobian matrix.
  void set_jacobian(const MatrixType& jacobian) { jacobian_ = jacobian; }

  /// @return Check if the cache has the integral mapping.
  bool has_mapping() const { return mapping_.has_value(); }
  /// @return The integral mapping.
  const TensorType& get_mapping() const { return *mapping_; }
  /// @param mapping Set the integral mapping.
  void set_mapping(const TensorType& mapping) { mapping_ = mapping; }

  /// @return Check whether the cache has the Jacobian inverse.
  bool has_jacobian_inverse() const { return jacobian_inverse_.has_value(); }
  /// @return The Jacobian inverse of shape (physical dimensions, parametric
  /// dimensions).
  const auto& get_jacobian_inverse() const { return *jacobian_inverse_; }
  /// @param jacobian_inverse Set the Jacobian inverse.
  void set_jacobian_inverse(const MatrixType& jacobian_inverse)
  {
    jacobian_inverse_ = jacobian_inverse;
  }

  /// @return Whether the cache has the mapped basis.
  bool has_mapped_basis() const { return mapped_basis_.has_value(); }
  /// @return Return the mapped basis.
  const TensorType& get_mapped_basis() const { return *mapped_basis_; }
  /// @param mapped_basis Set the mapped basis.
  void set_mapped_basis(const TensorType& mapped_basis)
  {
    mapped_basis_ = mapped_basis;
  }
};

// =================================================================
// Backend templates

/// @brief Backend for the discontinuous Galerkin assembly.
template<typename BlockType, typename ConfigType>
class DGBackend {
protected:
  // =================================================================
  // Protected data
  /// Configuration.
  const ConfigType* config_;
  /// Pointer to the specific mesh block.
  BlockType::Ptr block_;

public:
  // =================================================================
  // Public constructors
  DGBackend(const typename BlockType::Ptr& block, const ConfigType& config)
    : block_(block), config_(&config)
  {}

  // =================================================================
  // Public methods
  /// @brief Send the backend info to another device or data type (in-place).
  /// @param options The tensor options.
  void to_(const torch::TensorOptions& options)
  {
    block_ = block_->to(options);
  }

  // =================================================================
  // Public Getters / Setters
  /// @return The pointer to the mesh block.
  const BlockType::Ptr& get_block() const noexcept { return block_; }
  /// @return The configuration for assembly.
  const ConfigType& get_config() const noexcept { return *config_; }
};

/// @brief The backend for assembly of the multigroup discrete ordinates
/// first-order neutron transport operators with discontinuous IGA. This
/// assembles the operators into the typed format for a specific number of
/// dimensions.
template<FormatType Fmt, int64_t NumDim>
class DIGAFirstOrderTransportBackend
  : public DGBackend<cad::Patch, DGTransportAssemblerConfig> {
public:
  // =================================================================
  // Public types
  using Tensors = c10::SmallVector<torch::Tensor, 3>;
  using ReturnType = typename Return<Fmt, NumDim>::Type;
  using ReturnMatrixType = typename Return<Fmt, NumDim>::MatrixType;

protected:
  // =================================================================
  // Protected data
  /// Quadratures for each spatial dimension.
  math::ProductQuadrature::Ptr spatial_qset_;
  /// Evaluated quadrature points on all elements along a dimension.
  Tensors quad_points_;
  /// Angular quadrature.
  math::QuadratureSet::Ptr angular_qset_;
  /// Pointer to the material that fills this block.
  const xs::Material* material_ = nullptr;
  /// Caching struct for format specific data.
  BackendCache<cad::Patch, DGAssemblerConfig, Fmt, NumDim> cache_;

public:
  // =================================================================
  // Public constructors
  DIGAFirstOrderTransportBackend(const cad::Patch::Ptr& block,
    const math::QuadratureSet::Ptr& angular_qset,
    const xs::Server::Ptr& xs_server,
    const DGTransportAssemblerConfig& config = DGTransportAssemblerConfig())
    : DIGAFirstOrderTransportBackend(block, angular_qset,
        xs_server->get_material(block->get_fill_id()), config)
  {}

  DIGAFirstOrderTransportBackend(const cad::Patch::Ptr& block,
    const math::QuadratureSet::Ptr& angular_qset, const xs::Material& material,
    const DGTransportAssemblerConfig& config = DGTransportAssemblerConfig())
    : DGBackend<cad::Patch, DGTransportAssemblerConfig>(block, config),
      angular_qset_(angular_qset), material_(&material), cache_(block, config)
  {
    // Sanity check the number of dimensions
    TORCH_CHECK(block_->get_ndim() == NumDim,
      "CAD patch dimension mismatch with backend template configuration");

    // Reserve the number of dimensions
    quad_points_.reserve(NumDim);
    math::ProductQuadrature::Quads spatial_quads;

    auto options = c10::TensorOptions()
                     .device(block->get_device())
                     .dtype(block->get_dtype());

    // Iterate through each dimension
    for (const auto& basis : block_->get_basis()) {
      // Create a quadrature for this dimension
      spatial_quads.push_back(
        math::QuadratureSet1D::gauss_legendre(basis.get_degree() + 1));

      // Get the unique knots in the knot vector
      torch::Tensor unique_knots = basis.get_unique_knots().to(options);

      // Get the start and width of each span
      torch::Tensor span_starts = unique_knots.slice(0, 0, -1);
      torch::Tensor span_widths = unique_knots.diff();

      // Map parent element space to parametric space
      quad_points_.push_back(
        ((spatial_quads.back()->get_points().unsqueeze(0).to(options) + 1.0) *
            span_widths.unsqueeze(1) / 2.0 +
          span_starts.unsqueeze(1))
          .flatten());
    }

    spatial_qset_ = math::ProductQuadrature::create(spatial_quads);
  }

  // =================================================================
  // Public methods
  auto assemble_ordinates() -> Return<Fmt, NumDim>::VectorType;
  auto assemble_basis() -> ReturnType;
  auto assemble_basis_ders() -> Return<Fmt, NumDim + 1>::VectorType;
  auto assemble_scattering_kernel(
    std::optional<linalg::TTEngine> spatial = std::nullopt) const -> ReturnType;
  auto assemble_angular_integral() const -> ReturnType;
  auto assemble_jacobian() -> ReturnMatrixType;
  auto assemble_integral_mapping() -> ReturnType;
  auto assemble_jacobian_inverse() -> ReturnMatrixType;
  auto assemble_loss_operator() -> Return<Fmt, NumDim>::OperatorType;
  auto assemble_scatter_operator() -> Return<Fmt, NumDim>::OperatorType;
  auto assemble_fission_operator() -> Return<Fmt, NumDim>::OperatorType;
  auto assemble_boundary_operators(size_t dim, bool is_upper)
    -> std::tuple<typename Return<Fmt, NumDim>::OperatorType,
      typename Return<Fmt, NumDim>::OperatorType>;
  auto assemble_outflow_boundary_operator(const ReturnType& basis,
    const typename Return<Fmt, NumDim>::VectorType& normal,
    const ReturnType& mapping) -> ReturnType;
  auto assemble_inflow_boundary_operator(const ReturnType& basis,
    const typename Return<Fmt, NumDim>::VectorType& normal,
    const ReturnType& mapping, const BoundaryType condition)
    -> std::optional<ReturnType>;
  auto assemble_interface_boundary_operator(const ReturnType& basis,
    const typename Return<Fmt, NumDim>::VectorType& normal,
    const ReturnType& mapping, bool is_outflow) -> ReturnType;

  linalg::TTEngine apply_angular_weights(const linalg::TTEngine& op,
    const c10::SmallVector<size_t, 2>& core_idxs) const
  {
    if constexpr (NumDim > 1) {
      // Cast to derived class
      if (!angular_qset_->is_tensor_product()) {
        throw utils::runtime_error(
          "ttnte::physics::DIGAFirstOrderTransportBackend::apply_angular_"
          "weights",
          "The angular quadrature set must be a tensor product quadrature set");
      }
      auto angular_qset =
        std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);

      const auto& quads = angular_qset->get_quads();
      assert(quads.size() == 2 && core_idxs.size() == 2);
      assert(core_idxs[0] < op.size() && core_idxs[1] < op.size());
      assert(op[core_idxs[0]].size(2) == quads[0]->get_num_dofs());
      assert(op[core_idxs[1]].size(2) == quads[1]->get_num_dofs());

      auto cores = op.get_cores();
      auto& mu_core = cores[core_idxs[0]];
      auto& gamma_core = cores[core_idxs[1]];
      mu_core = mu_core * quads[0]->get_weights().reshape({1, 1, -1, 1});
      gamma_core = gamma_core * quads[1]->get_weights().reshape({1, 1, -1, 1});

      return linalg::TTEngine(cores, false);

    } else if constexpr (NumDim == 1) {
      assert(std::dynamic_pointer_cast<math::QuadratureSet1D>(angular_qset_));

      // Cast to derived class
      auto angular_qset =
        std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);

      assert(core_idxs.size() == 1);
      assert(angular_qset->get_num_dofs() == op[0].size(2));

      auto cores = op.get_cores();
      auto& mu_core = cores[core_idxs[0]];
      mu_core = mu_core * angular_qset->get_weights().reshape({1, 1, -1, 1});

      return linalg::TTEngine(cores, false);
    }

    throw utils::runtime_error(
      "ttnte::physics::DIGAFirstOrderTransportBackend<FormatType::TENSOR_"
      "TRAIN>"
      "::apply_angular_weights",
      "This function is not implemented for general angular quadrature sets");
  }

  /// In-place move object for the data type and device.
  DIGAFirstOrderTransportBackend& to_(const torch::TensorOptions& options)
  {
    DGBackend<cad::Patch, DGTransportAssemblerConfig>::to_(options);

    cache_.to_(options);
    spatial_qset_->to_(options);
    for (size_t i = 0; i < quad_points_.size(); i++) {
      quad_points_[i] = quad_points_[i].to(options);
    }

    return *this;
  }

  DIGAFirstOrderTransportBackend& to_(
    const torch::Device& device, const torch::ScalarType& dtype)
  {
    return to_(torch::TensorOptions().device(device).dtype(dtype));
  }

  // =================================================================
  // Public Getters / Setters
  const math::ProductQuadrature::Ptr& get_spatial_qset() const noexcept
  {
    return spatial_qset_;
  }
  const Tensors& get_quad_points() const noexcept { return quad_points_; }
  const math::QuadratureSet::Ptr& get_angular_qset() const noexcept
  {
    return angular_qset_;
  }
  const xs::Material& get_material() const noexcept { return *material_; }
};

} // namespace ttnte::physics::backends
