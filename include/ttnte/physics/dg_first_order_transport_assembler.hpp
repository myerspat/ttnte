#pragma once

#include "ttnte/cad/patch.hpp"
#include "ttnte/linalg/ops.hpp"
#include "ttnte/linalg/source.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include "ttnte/physics/dg_assembler.hpp"
#include "ttnte/physics/dg_first_order_transport_backends.hpp"
#include "ttnte/physics/particle_balance.hpp"
#include "ttnte/utils/exception.hpp"
#include <c10/util/SmallVector.h>
#include <memory>
#include <variant>
#include <vector>

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
  /// Outgoing partial-current reduction operator per face (only defined for
  /// INTERNAL faces -- see NeighborCoupling::current_op).
  BoundaryOps current_ops_;
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

    // Build the volumetric/MMS source, if one was attached to this block,
    // and any per-face prescribed-incident-flux contributions (folded into
    // the same RHS below, alongside the volumetric term).
    std::vector<linalg::State> source_terms;
    const auto& fixed_source_spec = this->block_->get_fixed_source();
    if (fixed_source_spec.has_value() && fixed_source_spec->defined()) {
      source_terms.push_back(std::visit(
        [&](auto* backend) {
          return backend->assemble_source(*fixed_source_spec);
        },
        get_backend_variant(this->config_.source_fmt)));
    }

    // Build outflow and inflow boundary operators
    c10::SmallVector<BoundaryType, 6> conditions;
    conditions.reserve(2 * NumDim);
    outflow_ops_.reserve(2 * NumDim);
    inflow_ops_.reserve(2 * NumDim);
    current_ops_.reserve(2 * NumDim);

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
        current_ops_.push_back(std::get<2>(boundary_tuple));

        // Save the boundary condition
        const auto& binfo = this->block_->get_boundary_info(dim, is_upper);
        conditions.push_back(binfo.get_type());

        // Prescribed incident-flux boundary: a fixed, known value, not this
        // patch's own unknown -- fold its contribution into the RHS once
        // here at assembly time (unlike INTERNAL couplings, it is never
        // re-exchanged over the course of the DD sweep).
        if (binfo.get_type() == BoundaryType::INCIDENT) {
          const auto& incident_spec = binfo.get_source();
          if (incident_spec.has_value() && incident_spec->defined()) {
            linalg::State incident_state = std::visit(
              [dim, is_upper, &incident_spec](auto* backend) {
                return backend->assemble_incident_source(
                  dim, is_upper, *incident_spec);
              },
              get_backend_variant(this->config_.source_fmt));

            linalg::State incident_rhs =
              linalg::mv(inflow_ops_.back(), incident_state);
            incident_rhs.round_(
              this->config_.rounding.eps, this->config_.rounding.max_rank);
            source_terms.push_back(std::move(incident_rhs));
          }
        }
      }
    }

    if (!source_terms.empty()) {
      source_ = source_terms.size() == 1 ? std::move(source_terms[0])
                                         : linalg::direct_sum(source_terms);
      source_.round_(
        this->config_.rounding.eps, this->config_.rounding.max_rank);
    }

    // Combine the operators on the left hand side. When
    // source_iterate_scattering is set, scattering is kept out of `lhs`
    // entirely and attached to the LinearSystem separately (below) for a
    // LocalSolver that iterates it explicitly (see
    // solvers::SourceIterationSolver) to consume -- see the
    // handles_scatter_source()/get_scatter_op() consistency guard.
    double inner_eps =
      this->config_.rounding.eps / static_cast<double>(2 * NumDim + 2);

    auto lhs = this->config_.source_iterate_scattering
                 ? interior_loss_op_
                 : interior_loss_op_ - scatter_op_;
    lhs.round_(inner_eps, this->config_.rounding.max_rank);

    assert(outflow_ops_.size() == inflow_ops_.size());
    for (size_t i = 0; i < outflow_ops_.size(); i++) {
      const auto& outflow_op = outflow_ops_[i];
      const auto& inflow_op = inflow_ops_[i];

      if (outflow_op.defined()) {
        lhs += outflow_op;
        lhs.round_(inner_eps, this->config_.rounding.max_rank);
      }
      // REFLECTIVE inflow: inflow_op is already albedo * B_refl (scaled
      // once, in the backend's assemble_boundary_operators()), so this is
      // LHS = ... + B_out - albedo * B_refl.
      if (inflow_op.defined() && conditions[i] != BoundaryType::INTERNAL &&
          conditions[i] != BoundaryType::INCIDENT) {
        lhs -= inflow_op;
        lhs.round_(inner_eps, this->config_.rounding.max_rank);
      }
    }
    lhs.round_(this->config_.rounding.eps, this->config_.rounding.max_rank);

    // Build couplings for each INTERNAL face before creating the linear system
    // so that boundary operators are packed into the contiguous pinned buffer.
    c10::SmallVector<linalg::NeighborCoupling, 6> couplings;
    {
      size_t face_idx = 0;
      for (int64_t dim = 0; dim < NumDim; dim++) {
        for (bool is_upper : {false, true}) {
          if (conditions[face_idx] == BoundaryType::INTERNAL) {
            for (const auto& conn :
              this->block_->get_boundary_info(dim, is_upper)
                .get_connections()) {
              linalg::NeighborCoupling coupling;
              coupling.fid = face_idx;
              coupling.connection = conn;
              coupling.boundary_op = inflow_ops_[face_idx];
              coupling.current_op = current_ops_[face_idx];
              coupling.dim = static_cast<size_t>(dim);
              coupling.is_upper = is_upper;
              coupling.recv_buffer = linalg::State();
              couplings.push_back(std::move(coupling));
            }
          }
          ++face_idx;
        }
      }
    }

    // A fixed source on a fissile fill would need combined subcritical-
    // multiplication support (fixed source + fission_op_ applied to the
    // current iterate every outer iteration, with no 1/k rescale) that
    // doesn't exist yet -- fail loudly here rather than silently dropping
    // the fixed source (EigenSource takes priority below) or silently
    // solving with a stale/never-updated EigenSource (solve_fixed_source()
    // never calls EigenSource::update()/scale(), those are only driven by
    // solve_eigenvalue()'s own loop).
    if (fission_op_.defined() && source_.defined()) {
      throw utils::runtime_error(error_context("assemble"),
        "A fixed source is attached to a block with a fissile fill. "
        "Combined fixed-source + fissile (subcritical multiplication) "
        "problems are not yet supported -- remove the fixed source or use "
        "a non-fissile fill.");
    }

    // Wrap fission operator in an EigenSource so the flat buffer carries it
    // to the device in one DMA transfer; nullptr for non-fissile problems.
    // A fixed source (volumetric and/or boundary-incident, accumulated into
    // source_ above) is wrapped in a plain Source instead -- its state is
    // already static, so to_buffer()/from_buffer() carry it through the same
    // flat-buffer DMA transfer with no per-iteration update.
    linalg::Source::Ptr source = nullptr;
    if (fission_op_.defined()) {
      source = linalg::EigenSource::create(fission_op_);
    } else if (source_.defined()) {
      source = linalg::Source::create(source_);
    }

    // Build the moment projector, if requested -- rides the interior
    // operator's own format since it's packed into the same flat buffer.
    linalg::Operator moment_projector;
    if (this->config_.assemble_moment_projector) {
      moment_projector = std::visit(
        [order = this->config_.moment_order](
          auto* backend) { return backend->assemble_moment_projector(order); },
        get_backend_variant(this->config_.interior_loss_fmt));
    }

    // Setup the linear system and return. scatter_op_ is only attached when
    // it was kept out of lhs above -- LinearSystem::get_scatter_op() stays
    // undefined otherwise (already folded into lhs, nothing more to do).
    this->linear_system_ =
      linalg::LinearSystem::create(lhs, std::move(couplings), linalg::State(),
        std::move(source), std::nullopt, std::move(moment_projector),
        this->config_.source_iterate_scattering ? scatter_op_
                                                : linalg::Operator());
    return this->linear_system_;
  }

  /// @brief Compute this patch's own particle-balance diagnostics from its
  /// current converged state. Purely local -- no data is shared between
  /// patches to build this (see TransportDriver::global_balance() for how
  /// INTERNAL faces' incoming currents get resolved across patches, the only
  /// place cross-patch data is used for this feature).
  /// @param psi The converged state to compute the balance from.
  /// @param eps TT-rounding tolerance for the balance/leakage functionals'
  /// own construction -- independent of config_.rounding.eps, so eps=0 gives
  /// an exact/benchmarking-grade result regardless of the solve's own
  /// tolerance.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return This patch's PatchBalance.
  PatchBalance compute_balance(
    const linalg::State& psi, double eps, int64_t max_rank) final override
  {
    const auto& material =
      xs_server_->get_material(this->block_->get_fill_id());
    int64_t num_groups = material.get_num_groups();
    auto options =
      torch::TensorOptions().device(psi.get_device()).dtype(psi.get_dtype());

    auto to_vector = [](const linalg::State& reduced) {
      return reduced.to_dense().flatten();
    };
    auto reduce = [&](const torch::Tensor& energy_matrix,
                    const linalg::State& state) {
      auto functional = std::visit(
        [&](auto* backend) {
          return backend->assemble_balance_functional(
            energy_matrix, eps, max_rank);
        },
        get_backend_variant(this->config_.source_fmt));
      return to_vector(linalg::mv(functional, state));
    };

    PatchBalance result;
    result.gid = this->block_->get_gid();

    result.absorption = reduce(torch::diag(material.get_absorption()), psi);
    result.scatter_out = reduce(
      torch::diag(material.get_total() - material.get_absorption()), psi);
    result.scatter_in =
      reduce(material.get_scatter_gtg().narrow(0, 0, 1).squeeze(0), psi);
    result.fission_source =
      reduce(material.is_fissile()
               ? torch::outer(material.get_chi(), material.get_nu_fission())
               : torch::zeros({num_groups, num_groups}, options),
        psi);
    // source_ is already a Galerkin-projected load vector (its own values
    // are ∫B_i(r)*density(r)dr, not raw field samples -- unlike psi), so its
    // total is a PLAIN sum over the spatial DOF axis (partition-of-unity),
    // not another one-sided load-vector contraction the way reduce() applies
    // to psi -- angular_qset_->integrate() first collapses angle, leaving
    // (spatial, energy) to sum over per group.
    if (source_.defined()) {
      linalg::State integrated =
        angular_qset_->integrate(source_, eps, max_rank);
      result.fixed_source =
        integrated.to_dense().reshape({-1, num_groups}).sum(0);
    } else {
      result.fixed_source = torch::zeros({num_groups}, options);
    }

    result.leakage = torch::zeros({num_groups}, options);
    result.faces.reserve(2 * NumDim);
    for (int64_t dim = 0; dim < NumDim; dim++) {
      for (bool is_upper : {false, true}) {
        const auto& binfo = this->block_->get_boundary_info(dim, is_upper);

        FaceBalance face;
        face.dim = static_cast<size_t>(dim);
        face.is_upper = is_upper;
        face.type = binfo.get_type();

        if (face.type == BoundaryType::DEGENERATE) {
          face.outgoing = torch::zeros({num_groups}, options);
          result.faces.push_back(std::move(face));
          continue;
        }

        auto leakage_functional = [&](bool is_outflow, bool narrowed_input) {
          return std::visit(
            [&](auto* backend) {
              return backend->assemble_leakage_functional(
                dim, is_upper, is_outflow, narrowed_input, eps, max_rank);
            },
            get_backend_variant(this->config_.outflow_fmt));
        };

        // Outgoing is always applied to the full, un-narrowed psi.
        face.outgoing =
          to_vector(linalg::mv(leakage_functional(true, false), psi));

        if (face.type == BoundaryType::REFLECTIVE) {
          // Incoming = the weak form's own inflow term tested against the
          // constant function: the (albedo-scaled, see
          // DGFirstOrderTransportBackend::assemble_boundary_operators())
          // reflective inflow operator applied to psi -- i.e.
          // albedo * mirrored outgoing angular flux -- integrated over angle
          // and summed over the spatial DOF axis (partition of unity), the
          // same reduction used for `fixed_source` above. This is exactly
          // the term subtracted in the assembled LHS, so the balance closes
          // to solve tolerance for any albedo in [0, 1] (albedo = 0 gives 0,
          // matching VACUUM), unlike psi's own incoming-direction trace,
          // which the DG discretization only matches weakly. Falls back to
          // albedo * outgoing (equal for a mirror-symmetric quadrature, which
          // the reflection permutation assumes) if the inflow operators are
          // not available (assemble() not called).
          const size_t face_idx =
            static_cast<size_t>(dim) * 2 + static_cast<size_t>(is_upper);
          if (face_idx < inflow_ops_.size() &&
              inflow_ops_[face_idx].defined()) {
            linalg::State inflow =
              linalg::mv(inflow_ops_[face_idx].to(options), psi);
            inflow.round_(eps, max_rank);
            linalg::State integrated =
              angular_qset_->integrate(inflow, eps, max_rank);
            face.incoming =
              integrated.to_dense().reshape({-1, num_groups}).sum(0);
          } else {
            face.incoming = binfo.albedo() * face.outgoing;
          }
          result.leakage += face.outgoing - *face.incoming;

        } else if (face.type == BoundaryType::INCIDENT) {
          const auto& incident_spec = binfo.get_source();
          if (incident_spec.has_value() && incident_spec->defined()) {
            linalg::State incident_state = std::visit(
              [&](auto* backend) {
                return backend->assemble_incident_source(
                  dim, is_upper, *incident_spec);
              },
              get_backend_variant(this->config_.source_fmt));
            // assemble_incident_source() narrows to a single point along
            // `dim` -- the incoming functional must match that convention.
            face.incoming = to_vector(
              linalg::mv(leakage_functional(false, true), incident_state));
            // Unlike REFLECTIVE, this face's `incoming` is NOT subtracted
            // from `leakage` -- it's the exact same physical quantity as
            // `fixed_source` (both come from the same prescribed source,
            // via two independent routes), so subtracting it here too would
            // double-count it. Only the outgoing/backscattered flux at this
            // face counts as leakage.
            result.leakage += face.outgoing;
          } else {
            result.leakage += face.outgoing;
          }

        } else if (face.type == BoundaryType::INTERNAL) {
          const auto& connections = binfo.get_connections();
          if (!connections.empty()) {
            face.neighbor_gid = connections[0].gid;
            face.neighbor_dim = connections[0].dim;
            face.neighbor_is_upper = connections[0].is_upper;
          }
          // Incoming (and this face's contribution to `leakage`) is resolved
          // across patches once every patch has been gathered -- see
          // TransportSolution::resolve_internal_faces().

        } else {
          // VACUUM: incoming is identically 0, not computed.
          result.leakage += face.outgoing;
        }

        result.faces.push_back(std::move(face));
      }
    }

    return result;
  }

  /// @brief Assemble a particle-balance reaction-rate functional (see
  /// DGFirstOrderTransportBackend::assemble_balance_functional()), computed
  /// on demand (not cached) and dispatched to this assembler's own backend
  /// (this->config_.interior_loss_fmt). `mv(functional, psi)` collapses
  /// space and angle to size 1 directly -- `.to_dense().reshape({num_groups})`
  /// is the per-group scalar result.
  /// @param energy_matrix (num_groups, num_groups) weight/transfer matrix --
  /// energy_matrix[i, j] is the output group i's coefficient on input group
  /// j (see DGFirstOrderTransportBackend::assemble_balance_functional()'s
  /// own doc for the diagonal-vs-transfer-matrix convention).
  /// @param eps TT-rounding tolerance for this functional's own
  /// construction, independent of config_->rounding.eps.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return The assembled reduction functional.
  linalg::Operator assemble_balance_functional(
    const torch::Tensor& energy_matrix, double eps, int64_t max_rank)
  {
    return std::visit(
      [&](auto* backend) {
        return backend->assemble_balance_functional(
          energy_matrix, eps, max_rank);
      },
      get_backend_variant(this->config_.interior_loss_fmt));
  }

  /// @brief Assemble a particle-balance leakage (partial-current) functional
  /// for one boundary face (see
  /// DGFirstOrderTransportBackend::assemble_leakage_functional()), computed
  /// on demand (not cached). Unlike `current_ops_`/`get_current_ops()`
  /// (built for the per-point Schwarz convergence check -- angular cores
  /// folded away only, spatial/energy structure along the face left intact),
  /// this integrates over the face's spatial extent too, giving a scalar
  /// per group.
  /// @param dim The dimension of the face.
  /// @param is_upper Whether the face is at the upper or lower end of `dim`.
  /// @param is_outflow True for the outgoing ((Omega.n)_+) functional, false
  /// for incoming ((Omega.n)_-).
  /// @param narrowed_input Whether this will be applied to a State already
  /// narrowed to a single point along `dim`, rather than the full state
  /// (e.g. psi itself -- the usual case for the outgoing functional).
  /// @param eps TT-rounding tolerance, independent of config_->rounding.eps.
  /// @param max_rank TT-rounding max rank, paired with `eps`.
  /// @return The assembled leakage functional.
  linalg::Operator assemble_leakage_functional(size_t dim, bool is_upper,
    bool is_outflow, bool narrowed_input, double eps, int64_t max_rank)
  {
    return std::visit(
      [&](auto* backend) {
        return backend->assemble_leakage_functional(
          dim, is_upper, is_outflow, narrowed_input, eps, max_rank);
      },
      get_backend_variant(this->config_.interior_loss_fmt));
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
  /// @return The outgoing partial-current reduction operators for each
  /// boundary (only defined for INTERNAL faces).
  const BoundaryOps& get_current_ops() const noexcept { return current_ops_; }
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
