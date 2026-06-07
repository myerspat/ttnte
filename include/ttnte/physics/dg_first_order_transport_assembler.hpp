#pragma once

#include "ttnte/cad/patch.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include "ttnte/physics/dg_assembler.hpp"
#include "ttnte/physics/dg_first_order_transport_backends.hpp"
#include "ttnte/utils/exception.hpp"
#include <c10/util/SmallVector.h>
#include <memory>
#include <variant>

namespace ttnte::physics {

/// @brief Assembler for first-order neutron transport with multigroup and
/// discrete ordinates for tensor product mesh blocks.
template<typename BlockType, int64_t NumDim>
class DGFirstOrderTransportAssembler
  : public DGAssembler<BlockType, DGTransportAssemblerConfig> {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<DGFirstOrderTransportAssembler>;
  using BoundaryOps = c10::SmallVector<linalg::Operator, 6>;

  // Backend types for both formats
  using DenseBackend = backends::DGFirstOrderTransportBackend<BlockType,
    FormatType::DENSE, NumDim>;
  using TTBackend = backends::DGFirstOrderTransportBackend<BlockType,
    FormatType::TENSOR_TRAIN, NumDim>;
  using BackendVariant = std::variant<DenseBackend*, TTBackend*>;

protected:
  // =================================================================
  // Protected data
  /// Interior loss operator (streaming + collision).
  linalg::Operator interior_loss_op_;
  /// Scattering operator.
  linalg::Operator scatter_op_;
  /// Fission operator.
  linalg::Operator fission_op_;
  /// Outflow boundary operator.
  BoundaryOps outflow_ops_;
  /// Inflow boundary operator.
  BoundaryOps inflow_ops_;
  /// Source vector for fixed source problems.
  linalg::State source_;

  /// Pointer to the angular quadrature set.
  math::QuadratureSet::Ptr angular_qset_;
  /// Pointer to the XS server.
  xs::Server::Ptr xs_server_;

  /// Lazy initialization of various backends
  std::unique_ptr<DenseBackend> dense_backend_;
  std::unique_ptr<TTBackend> tt_backend_;

  // =================================================================
  // Protected constructors
  DGFirstOrderTransportAssembler(const mesh::MeshBlock<BlockType>::Ptr& block,
    const math::QuadratureSet::Ptr& angular_qset,
    const xs::Server::Ptr& xs_server, const DGTransportAssemblerConfig& config)
    : DGAssembler<cad::Patch, DGTransportAssemblerConfig>(block, config),
      angular_qset_(angular_qset), xs_server_(xs_server)
  {}

  // =================================================================
  // Protected methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::linalg::DGFirstOrderTransportAssembler::" + func_name;
  }

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new DGFirstOrderTransportAssembler(std::forward<Args>(args)...));
  }

  /// @brief Get the backend standard variant.
  /// @param fmt The desired backend format.
  /// @return The backend for that specific format.
  BackendVariant get_backend_variant(FormatType fmt)
  {
    switch (fmt) {
    case FormatType::DENSE:
      return &get_backend<FormatType::DENSE>();
    case FormatType::TENSOR_TRAIN:
      return &get_backend<FormatType::TENSOR_TRAIN>();
    default:
      throw utils::runtime_error(
        error_context("get_backend_variant"), "This backend is not supported");
    }
  }

  /// @return The assembler backend for this format.
  template<FormatType Fmt>
  auto& get_backend()
  {
    if constexpr (Fmt == FormatType::DENSE) {
      if (!dense_backend_) {
        dense_backend_ = std::make_unique<DenseBackend>(
          this->block_, angular_qset_, xs_server_, this->config_);
      }
      return *dense_backend_;

    } else if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
      if (!tt_backend_) {
        tt_backend_ = std::make_unique<TTBackend>(
          this->block_, angular_qset_, xs_server_, this->config_);
      }
      return *tt_backend_;
    }
  }

  /// @brief Assemble the linear system for this mesh block for first-order
  /// neutron transport with multigroup and discrete ordinates.
  /// @return The full linear system object.
  linalg::LinearSystem::Ptr assemble() final override
  {
    // Build loss operator
    interior_loss_op_ = std::visit(
      [](auto* backend) { return backend->assemble_loss_operator(); },
      get_backend_variant(this->config_.interior_loss_fmt));

    // Build scattering operator
    scatter_op_ = std::visit(
      [](auto* backend) { return backend->assemble_scatter_operator(); },
      get_backend_variant(this->config_.scatter_fmt));

    // Build fission operator
    fission_op_ = std::visit(
      [](auto* backend) { return backend->assemble_fission_operator(); },
      get_backend_variant(this->config_.fission_fmt));

    // TODO: Add something for fixed source problems when the time comes

    // Build outflow and inflow boundary operators
    c10::SmallVector<BoundaryType, 6> conditions;
    conditions.reserve(2 * NumDim);
    outflow_ops_.reserve(2 * NumDim);
    inflow_ops_.reserve(2 * NumDim);

    for (int64_t dim = 0; dim < NumDim; dim++) {
      for (bool is_upper : {false, true}) {
        // Get both boundaries
        auto boundary_tuple = std::visit(
          [dim, is_upper](auto* backend) {
            return backend->assemble_boundary_operators(dim, is_upper);
          },
          get_backend_variant(this->config_.outflow_fmt));

        // Pass them to outflow and inflow
        outflow_ops_.push_back(std::get<0>(boundary_tuple));
        inflow_ops_.push_back(std::get<1>(boundary_tuple));

        // Save the boundary condition
        conditions.push_back(
          this->block_->get_boundary_info(dim, is_upper).get_type());
      }
    }

    // Combine the operators on the left hand side
    double inner_eps =
      this->config_.rounding.eps / static_cast<double>(2 * NumDim + 2);

    auto lhs = interior_loss_op_ - scatter_op_;
    lhs.round_(inner_eps, this->config_.rounding.max_rank);

    assert(outflow_ops_.size() == inflow_ops_.size());
    for (size_t i = 0; i < outflow_ops_.size(); i++) {
      const auto& outflow_op = outflow_ops_[i];
      const auto& inflow_op = inflow_ops_[i];

      if (outflow_op.defined()) {
        lhs += outflow_op;
        lhs.round_(inner_eps, this->config_.rounding.max_rank);
      }
      if (inflow_op.defined() && conditions[i] != BoundaryType::INTERNAL) {
        lhs -= inflow_op;
        lhs.round_(inner_eps, this->config_.rounding.max_rank);
      }
    }
    lhs.round_(this->config_.rounding.eps, this->config_.rounding.max_rank);

    // Setup the linear system and return
    this->linear_system_ = linalg::LinearSystem::create(lhs);
    return this->linear_system_;
  }

  // =================================================================
  // Public getters / setters
  /// @return The assembled interior loss operator.
  const linalg::Operator& get_interior_loss_op() const noexcept
  {
    return interior_loss_op_;
  }
  /// @return The assembled interior scattering operator.
  const linalg::Operator& get_scatter_op() const noexcept
  {
    return scatter_op_;
  }
  /// @return The assembled interior fission operator.
  const linalg::Operator& get_fission_op() const noexcept
  {
    return fission_op_;
  }
  /// @return The outflow boundary operators for each boundary.
  const BoundaryOps& get_outflow_ops() const noexcept { return outflow_ops_; }
  /// @return The inflow boundary operators for each boundary.
  const BoundaryOps& get_inflow_ops() const noexcept { return inflow_ops_; }
  /// @return Get the fixed source.
  const linalg::State& get_source() const noexcept { return source_; }

  /// @return The angular quadrature set.
  const math::QuadratureSet::Ptr& get_angular_qset() const noexcept
  {
    return angular_qset_;
  }
  /// @return The XS data.
  const xs::Server::Ptr& get_xs_server() const noexcept { return xs_server_; }
};

} // namespace ttnte::physics
