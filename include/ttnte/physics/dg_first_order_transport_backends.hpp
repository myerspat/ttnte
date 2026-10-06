#pragma once

#include "ttnte/cad/patch.hpp"
#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/state.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include "ttnte/physics/boundary_types.hpp"
#include "ttnte/physics/fixed_source.hpp"
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
};

/// @brief Return types for the tensor train format.
template<int64_t NumDim>
struct Return<FormatType::TENSOR_TRAIN, NumDim> {
  using Type = linalg::TTEngine;
  using VectorType = c10::SmallVector<linalg::TTEngine, NumDim>;
  using MatrixType = std::array<std::array<linalg::TTEngine, NumDim>, 3>;
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

/// @brief Primary template for the first-order discontinuous Galerkin transport
/// backend.
template<typename BlockType, FormatType Fmt, int64_t NumDim>
class DGFirstOrderTransportBackend;

/// @brief The backend for assembly of the multigroup discrete ordinates
/// first-order neutron transport operators with discontinuous IGA. This
/// assembles the operators into the typed format for a specific number of
/// dimensions.
template<FormatType Fmt, int64_t NumDim>
class DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>
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

private:
  // =================================================================
  // Private methods
  /// @brief Shared initialization of spatial_qset_/quad_points_ from the
  /// block's own basis -- purely geometric, no material/xs dependency.
  void init_spatial_quadrature_()
  {
    // Sanity check the number of dimensions
    TORCH_CHECK(block_->get_ndim() == NumDim,
      "CAD patch dimension mismatch with backend template configuration");

    // Reserve the number of dimensions
    quad_points_.reserve(NumDim);
    math::ProductQuadrature::Quads spatial_quads;

    auto options = c10::TensorOptions()
                     .device(block_->get_device())
                     .dtype(block_->get_dtype());

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

  /// @brief Build the raw (Omega . n)_+/- upwind mask. For NumDim == 1 this
  /// is angular-only. For NumDim > 1 it also carries the face's own
  /// tangential-dimension dependence (the outward normal varies pointwise
  /// along a curved IGA boundary), evaluated at FACE QUADRATURE points --
  /// i.e. cores [2, NumDim+1) are NOT yet expressed in the DOF/control-point
  /// basis; see project_boundary_mask() for that step. Factored out of
  /// assemble_interface_boundary_operator() (which continues on to project
  /// this against the DG basis and integrate over the face) so that method
  /// and assemble_boundary_operators()'s current-reduction operator (which
  /// instead just folds in the angular quadrature weights, for
  /// LocalSolver::postsolve()'s Schwarz convergence check) share this
  /// construction without duplicating the ordinate/TT-cross logic.
  /// @param normal Outward normal at this boundary face, per dimension --
  /// varies pointwise along the face for a curved IGA patch, not a single
  /// fixed direction.
  /// @param is_outflow True for (Omega . n)_+ (outflow), false for
  /// (Omega . n)_- (inflow).
  auto assemble_upwind_mask(
    const typename Return<Fmt, NumDim>::VectorType& normal, bool is_outflow)
    -> ReturnType;

  /// @brief Project a mask's tangential (face-quadrature) cores -- indices
  /// [2, NumDim+1) -- onto the DG basis, turning them into proper DOF-space
  /// (m=n=ctrlpts) matrix cores. This is the exact treatment
  /// assemble_interface_boundary_operator() applies when building B_out/B_in;
  /// factored out here so assemble_boundary_operators()'s current-reduction
  /// operator can apply the identical projection to its own angular-weighted
  /// mask without duplicating the einsum logic (only NumDim > 1 has
  /// tangential cores to project at all -- see assemble_upwind_mask()).
  /// Angular cores (indices 0, 1) are left untouched.
  /// @param mask Angular(+face-tangential-quadrature) mask -- e.g. the raw
  /// output of assemble_upwind_mask(), optionally with apply_angular_weights()
  /// already applied.
  /// @param basis Per-tangential-dim DG basis matrices.
  /// @param mapping Per-tangential-dim quadrature-weighted affine mapping.
  auto project_boundary_mask(linalg::TTEngine mask, const ReturnType& basis,
    const ReturnType& mapping) -> ReturnType;

  /// @brief Compute a boundary face's tangential DG basis, pointwise outward
  /// normal (accounting for IGA/NURBS curvature via the Jacobian cross
  /// product), and quadrature/Jacobian mapping. Factored out of
  /// assemble_boundary_operators() so it can be shared, unchanged, with
  /// assemble_leakage_functional() (the balance diagnostic) without
  /// duplicating the curved-boundary normal derivation. NumDim > 1 only --
  /// NumDim == 1 has no tangential dimension and a trivial (sign-only)
  /// normal, built inline at each call site instead.
  /// @param dim The dimension of the face.
  /// @param is_upper Whether the face is at the upper or lower end of `dim`.
  /// @return (tangential basis, per-dimension outward normal, quadrature/
  /// Jacobian mapping).
  auto assemble_boundary_geometry(size_t dim, bool is_upper)
    -> std::tuple<ReturnType, typename Return<Fmt, NumDim>::VectorType,
      ReturnType>;

public:
  // =================================================================
  // Public constructors
  DGFirstOrderTransportBackend(const cad::Patch::Ptr& block,
    const math::QuadratureSet::Ptr& angular_qset,
    const xs::Server::Ptr& xs_server,
    const DGTransportAssemblerConfig& config = DGTransportAssemblerConfig())
    : DGFirstOrderTransportBackend(block, angular_qset,
        xs_server->get_material(block->get_fill_id()), config)
  {}

  DGFirstOrderTransportBackend(const cad::Patch::Ptr& block,
    const math::QuadratureSet::Ptr& angular_qset, const xs::Material& material,
    const DGTransportAssemblerConfig& config = DGTransportAssemblerConfig())
    : DGBackend<cad::Patch, DGTransportAssemblerConfig>(block, config),
      angular_qset_(angular_qset), material_(&material), cache_(block, config)
  {
    init_spatial_quadrature_();
  }

  /// @brief Construct a backend with no material -- only valid for the
  /// purely-geometric methods (assemble_basis(), assemble_integral_mapping(),
  /// assemble_jacobian(), ...) that never touch get_material(). Intended for
  /// cheaply recomputing spatial integration weights on demand (e.g.
  /// TransportSolution::compute_errors()) without needing an
  /// xs::Server/Material.
  DGFirstOrderTransportBackend(const cad::Patch::Ptr& block,
    const math::QuadratureSet::Ptr& angular_qset,
    const DGTransportAssemblerConfig& config = DGTransportAssemblerConfig())
    : DGBackend<cad::Patch, DGTransportAssemblerConfig>(block, config),
      angular_qset_(angular_qset), material_(nullptr), cache_(block, config)
  {
    init_spatial_quadrature_();
  }

  /// @brief Construct a backend with no material and EXTERNALLY supplied
  /// quadrature points, bypassing init_spatial_quadrature_()'s
  /// self-derivation from this block's own knot vector. Only valid for
  /// assemble_basis() (spatial_qset_ is left unset, so
  /// assemble_integral_mapping() and anything depending on it is not valid
  /// on an instance built this way). Intended for evaluating a DIFFERENT
  /// patch's basis -- e.g. a differently-refined (more knot spans and/or
  /// higher polynomial degree) reference solution sharing the same
  /// parametric domain -- at another patch's own quadrature points, so the
  /// two can be compared pointwise (see
  /// TransportSolution::compute_errors()).
  DGFirstOrderTransportBackend(const cad::Patch::Ptr& block,
    const math::QuadratureSet::Ptr& angular_qset, Tensors quad_points,
    const DGTransportAssemblerConfig& config = DGTransportAssemblerConfig())
    : DGBackend<cad::Patch, DGTransportAssemblerConfig>(block, config),
      angular_qset_(angular_qset), material_(nullptr), cache_(block, config)
  {
    TORCH_CHECK(block_->get_ndim() == NumDim,
      "CAD patch dimension mismatch with backend template configuration");
    quad_points_ = std::move(quad_points);
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
  linalg::Operator assemble_loss_operator();
  linalg::Operator assemble_scatter_operator();
  linalg::Operator assemble_fission_operator();
  /// @brief Assemble a fixed-source RHS vector (space x angle x energy) from
  /// a FixedSource source: an isotropic contribution (physical-unit strength,
  /// used directly as the uniform per-direction value -- the angular
  /// quadrature's weights are always normalized to sum to 1, not to
  /// weighting_factor(), so no extra normalization is needed), an arbitrary
  /// `function(coords)` contribution evaluated at physical-space/angle/
  /// energy sample points via TT-cross, an already-projected spatially
  /// varying load (`projected_isotropic_source`, compressed to a TT and
  /// broadcast uniformly across ordinates), or any combination (summed via
  /// direct_sum).
  /// @param source The source specification for this block.
  /// @return The assembled source State.
  /// @throws utils::runtime_error If `projected_isotropic_source` does not
  /// have shape (control points per parametric dimension..., energy groups).
  linalg::State assemble_source(const FixedSource& source);
  /// @brief Assemble a prescribed incident-flux RHS contribution for one
  /// boundary face (`BoundaryType::INCIDENT`). Builds a raw nodal (not
  /// basis-projected) State over the tangential control points, narrowed to
  /// a single point along `dim` -- the same convention a NeighborCoupling's
  /// recv_buffer uses -- so it can be applied through
  /// `assemble_boundary_operators()`'s own inflow operator exactly the way
  /// the DAG's apply task applies `coupling.boundary_op` to a received
  /// neighbor State (the Galerkin test-function integration happens inside
  /// that operator, not here).
  /// @param dim The dimension of the face.
  /// @param is_upper Whether the face is at the upper or lower end of `dim`.
  /// @param source The source specification for this face. Only
  /// `isotropic_strength` is currently supported; `function` and
  /// `projected_isotropic_source` throw.
  /// @return The assembled incident-source State.
  linalg::State assemble_incident_source(
    size_t dim, bool is_upper, const FixedSource& source);
  std::tuple<linalg::Operator, linalg::Operator, linalg::Operator>
  assemble_boundary_operators(size_t dim, bool is_upper);
  /// @brief Assemble a particle-balance reaction-rate functional: a one-sided
  /// spatial load vector (not the bilinear mass matrix
  /// assemble_fission_operator()/assemble_scatter_operator() use, since this
  /// is a scalar functional of the state, not a reusable DOF-space operator)
  /// x a bare (un-expanded) angular-quadrature-weight reduction x an energy
  /// weight/transfer matrix. Applying mv(functional, psi) collapses space and
  /// angle to size 1 directly -- to_dense().reshape({num_groups}) is the
  /// final per-group result, no further reduction needed.
  /// @param energy_matrix (num_groups, num_groups) weight -- diagonal for a
  /// per-group-only reaction rate (absorption, scatter-out), or a full
  /// transfer matrix for group-to-group redistribution (scatter-in via the
  /// isotropic/l=0 moment only, fission source via chi (x) nu_fission).
  /// @param eps TT-rounding tolerance for this functional's own construction
  /// -- independent of config_->rounding.eps, so eps=0 gives a lossless
  /// functional for exact/benchmarking use regardless of the solve's own
  /// tolerance.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return The assembled reduction functional.
  linalg::Operator assemble_balance_functional(
    const torch::Tensor& energy_matrix, double eps, int64_t max_rank);
  /// @brief Assemble a particle-balance leakage (partial-current) functional
  /// for one boundary face, independent of assemble_boundary_operators() (no
  /// changes to the real PDE boundary operators). Built for any
  /// BoundaryType, using only local geometry -- no cross-patch data.
  /// @param dim The dimension of the face.
  /// @param is_upper Whether the face is at the upper or lower end of `dim`.
  /// @param is_outflow True for the outgoing ((Omega . n)_+) functional,
  /// false for incoming ((Omega . n)_-).
  /// @param narrowed_input Whether this functional will be applied to a
  /// State already narrowed to a single point along `dim` (e.g.
  /// assemble_incident_source()'s output, the same convention a
  /// NeighborCoupling's recv_buffer uses) rather than the full, un-narrowed
  /// state (e.g. psi itself, for the outgoing functional or REFLECTIVE's
  /// self-referential incoming).
  /// @param eps TT-rounding tolerance, independent of config_->rounding.eps.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return The assembled leakage functional.
  linalg::Operator assemble_leakage_functional(size_t dim, bool is_upper,
    bool is_outflow, bool narrowed_input, double eps, int64_t max_rank);
  /// @brief Assemble an orthogonal projector onto the low-order angular
  /// moments (scalar flux + current) of the angular flux, expressed back in
  /// full angular resolution -- i.e. `mv(P, psi)` directly gives psi's
  /// macroscopic part, no separate reduce/expand step needed. A pure
  /// interior/volume operator: identity over space and energy (the state's
  /// own spatial/energy representation is preserved exactly, not projected
  /// through the DG basis the way assemble_scattering_kernel()'s bilinear
  /// kernel is), so unlike assemble_balance_functional()/
  /// assemble_leakage_functional() this needs no boundary geometry.
  /// @param order Highest angular moment order to preserve. Only `order = 1`
  /// (scalar flux [P0] + full current vector [P1x/P1y/P1z, or just P1z for
  /// NumDim == 1]) is implemented; other values throw. Reserved so an
  /// arbitrary order (via the same associated-Legendre machinery
  /// assemble_scattering_kernel() already uses) is a natural future
  /// extension.
  /// @return The assembled moment projector.
  linalg::Operator assemble_moment_projector(int64_t order = 1);
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
  DGFirstOrderTransportBackend& to_(const torch::TensorOptions& options)
  {
    DGBackend<cad::Patch, DGTransportAssemblerConfig>::to_(options);

    cache_.to_(options);
    spatial_qset_->to_(options);
    for (size_t i = 0; i < quad_points_.size(); i++) {
      quad_points_[i] = quad_points_[i].to(options);
    }

    return *this;
  }

  DGFirstOrderTransportBackend& to_(
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
