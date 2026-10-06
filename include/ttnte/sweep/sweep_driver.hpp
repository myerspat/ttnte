#pragma once

#include "ttnte/driver/transport_solution.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/physics/boundary_types.hpp"
#include "ttnte/sweep/angular_moments.hpp"
#include "ttnte/sweep/patch_discretization.hpp"
#include "ttnte/sweep/sweep_config.hpp"
#include "ttnte/sweep/sweep_graph.hpp"
#include "ttnte/sweep/sweep_kernel.hpp"
#include "ttnte/task/task_graph.hpp"
#include "ttnte/task/task_scheduler.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/xs/server.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace ttnte::sweep {

/// @brief Discrete-ordinates transport sweep solver for discontinuous
/// multi-patch IGA.
///
/// Each local mesh block is a sweep unit: a whole patch for the patch
/// discontinuous (PDG) scheme, or a single element for the fully
/// discontinuous (FDG) scheme (a mesh whose patches were split into Bezier
/// elements). For every angle set of the SweepGraph and every group set, the
/// units are grouped into chunks (one unit, the whole subdomain, or a custom
/// grouping such as the parent patches of FDG elements; see
/// SweepGraph::schedule()). One scheduler task per chunk sweeps its units in
/// upwind order with the raw-array kernel sweep_unit() and passes outgoing
/// face traces downstream, so the DAG on the existing task scheduler has a
/// few coarse tasks rather than one per element. Source iteration on the
/// flux moments closes the scattering (and external) source.
///
/// The spatial and angular discretization is the one the TT solver
/// assembles (same quadratures, scaled normals, upwind masks, reflective
/// mapping and scattering kernel), so the PDG sweeper converges to the same
/// discrete solution.
template<typename BlockType, int64_t NumDim>
class SweepDriver {
public:
  // =================================================================
  // Public types
  using Mesh = mesh::Mesh<BlockType>;
  using Ptr = std::shared_ptr<SweepDriver>;
  using Solution = driver::TransportSolution<BlockType>;

private:
  // =================================================================
  // Private types
  /// Where a face's incoming angular flux comes from.
  struct FaceLink {
    physics::BoundaryType type = physics::BoundaryType::VACUUM;
    /// INTERNAL: neighbor local index and its face (2 * dim + is_upper).
    int64_t neighbor = -1;
    int64_t neighbor_face = -1;
    /// INTERNAL: (Nf) neighbor-face index of each of this face's dofs.
    torch::Tensor trace_map;
    /// INTERNAL: SweepInterface index and whether this patch is its first.
    int64_t interface = -1;
    bool is_first = false;
    /// REFLECTIVE: (N_Omega) reflected direction of each direction.
    torch::Tensor reflected;
    /// REFLECTIVE: specular albedo.
    double albedo = 1.0;
    /// Whether this face stores a trace buffer (read by a neighbor or by
    /// its own reflection).
    bool has_buffer = false;
  };

  /// Per-patch solver state.
  struct PatchState {
    int64_t gid = -1;
    typename BlockType::Ptr block;
    PatchDiscretization::Ptr disc;
    /// (G) total cross section, (L, G, G) scattering matrices.
    torch::Tensor sigma_t;
    torch::Tensor scatter;
    /// (G, N) external source load vector (undefined if none).
    torch::Tensor external;
    std::vector<FaceLink> links;
    /// Per face (N_Omega, G, Nf) outgoing traces of this and the previous
    /// sweep (undefined if the face has no buffer).
    std::vector<torch::Tensor> trace_current;
    std::vector<torch::Tensor> trace_previous;
    /// (num_moments, G, N) moments, the moments being accumulated by the
    /// running sweep, and the scattering + external source moments.
    torch::Tensor moments;
    torch::Tensor moments_next;
    torch::Tensor source_moments;
    /// (N_Omega, G, N) angular flux, if stored.
    torch::Tensor psi;
    /// Factorization cache (see SweepConfig::cache_factorizations).
    torch::Tensor factors;
    torch::Tensor factor_pivots;
    bool factors_ready = false;
    /// Guards moments_next accumulation across concurrent angle sets.
    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();
  };

  // =================================================================
  // Private data
  typename Mesh::Ptr mesh_;
  xs::Server::Ptr xs_server_;
  parallel::Communicator comm_;
  math::QuadratureSet::Ptr qset_;
  SweepConfig config_;

  /// (N_Omega, NumDim) streaming directions, (N_Omega, 3) unit vectors and
  /// (N_Omega) weights.
  torch::Tensor omega_;
  torch::Tensor omega3_;
  torch::Tensor weights_;
  /// (N_Omega, num_moments) harmonics, the same times the weights, and the
  /// Legendre order of each moment.
  torch::Tensor harmonics_;
  torch::Tensor weighted_harmonics_;
  torch::Tensor moment_order_;
  std::vector<int64_t> moment_order_raw_;
  int64_t num_groups_ = 0;

  std::vector<PatchState> patches_;
  std::unordered_map<int64_t, size_t> gid_to_local_;
  SweepGraph::Ptr graph_;
  std::vector<torch::Tensor> group_sets_;
  /// Directions of each angle set and groups of each group set as plain
  /// index vectors (read by the kernel).
  std::vector<std::vector<int64_t>> set_directions_;
  std::vector<std::vector<int64_t>> group_indices_;
  /// Chunk id of each local unit, and the task schedule of each angle set.
  std::vector<int64_t> chunk_of_;
  std::vector<std::vector<SweepChunk>> schedules_;

  task::TaskGraph dag_;
  std::unique_ptr<task::TaskScheduler> scheduler_;
  bool is_setup_ = false;
  /// Whether any local material is fissile.
  bool has_fission_ = false;
  /// Cumulative time spent inside sweep task bodies (all threads).
  std::atomic<int64_t> task_nanoseconds_ {0};

  /// Convergence history of the last solve.
  std::vector<double> errors_;
  std::vector<double> sweep_times_;
  int64_t num_iterations_ = 0;
  bool converged_ = false;

  // =================================================================
  // Private helpers
  [[nodiscard]] static std::string ctx(const std::string& f)
  {
    return "ttnte::sweep::SweepDriver::" + f;
  }

  /// @return (Nf) index of the neighbor face dof for each dof of this face,
  /// mirroring NeighborCoupling::set_recv_buffer() (permute, then flip the
  /// perpendicular dims).
  torch::Tensor build_trace_map(size_t dim, const mesh::NeighborInfo& conn,
    const PatchDiscretization& nb) const
  {
    const auto& nb_face = nb.face(conn.dim, conn.is_upper);
    if (NumDim == 1) {
      return torch::zeros({1}, torch::kInt64);
    }
    const auto& nb_shape = nb.shape();
    auto grid = torch::arange(nb.num_dofs(), torch::kInt64).reshape(nb_shape);
    auto face =
      grid.narrow(conn.dim, conn.is_upper ? nb_shape[conn.dim] - 1 : 0, 1);
    std::vector<int64_t> perm(
      conn.mapping.perm.begin(), conn.mapping.perm.end());
    bool needs_perm = false;
    for (size_t k = 0; k < perm.size(); k++)
      needs_perm |= perm[k] != static_cast<int64_t>(k);
    if (needs_perm)
      face = face.permute(perm);
    std::vector<int64_t> flip_dims;
    size_t perp = 0;
    for (int64_t d = 0; d < NumDim; d++) {
      if (d == static_cast<int64_t>(dim))
        continue;
      if (perp < conn.mapping.flip.size() && conn.mapping.flip[perp])
        flip_dims.push_back(d);
      perp++;
    }
    if (!flip_dims.empty())
      face = face.flip(flip_dims);

    auto volume_index = face.flatten();
    auto lookup = torch::full({nb.num_dofs()}, -1, torch::kInt64);
    lookup.index_put_(
      {nb_face.dofs}, torch::arange(nb_face.num_dofs(), torch::kInt64));
    auto map = lookup.index({volume_index});
    if ((map < 0).any().template item<bool>()) {
      throw utils::runtime_error(ctx("setup"),
        "Interface mapping does not land on the neighbor's face (non-"
        "conforming interface?)");
    }
    return map;
  }

  /// @return (N_Omega) index of the specular reflection of each direction on
  /// an axis-aligned face.
  torch::Tensor build_reflection(const FaceGeometry& face) const
  {
    auto mean = (face.normal * face.weights.unsqueeze(1)).sum(0);
    int64_t axis = mean.abs().argmax().template item<int64_t>();
    auto unit = face.normal / face.normal.norm(2, 1, true);
    auto off_axis =
      (unit.abs() - torch::one_hot(torch::tensor(axis), NumDim).to(unit))
        .abs()
        .max()
        .template item<double>();
    if (off_axis > 1e-6) {
      throw utils::runtime_error(
        ctx("setup"), "Reflective boundaries must be axis aligned");
    }
    auto reflected = omega3_.clone();
    reflected.select(1, axis).neg_();
    torch::Tensor mismatch, index;
    if (NumDim == 1) {
      // Slabs carry only mu
      auto dist = (reflected.select(1, 0).unsqueeze(1) -
                   omega3_.select(1, 0).unsqueeze(0))
                    .abs();
      std::tie(mismatch, index) = dist.min(1);
    } else {
      // Unit vectors: the reflection's match has Omega . Omega* = 1 (dot
      // products avoid cdist's cancellation-prone matmul path)
      auto dots = torch::matmul(reflected, omega3_.transpose(0, 1));
      torch::Tensor max_dot;
      std::tie(max_dot, index) = dots.max(1);
      mismatch = 1.0 - max_dot;
    }
    if (mismatch.max().template item<double>() > 1e-12) {
      throw utils::runtime_error(ctx("setup"),
        "The angular quadrature is not symmetric about a reflective face");
    }
    return index;
  }

  /// @brief Inflow of face `f` of unit `local` for an angle set.
  FaceInflow face_inflow(const AngleSet& set, size_t local, size_t f) const
  {
    const auto& P = patches_[local];
    const auto& link = P.links[f];
    FaceInflow in;
    if (link.type == physics::BoundaryType::INTERNAL) {
      const EdgeState state = link.is_first ? set.backward[link.interface]
                                            : set.forward[link.interface];
      if (state == EdgeState::NONE)
        return in;
      const auto& nb = patches_[link.neighbor];
      const auto& buffer = state == EdgeState::DEPENDENCY
                             ? nb.trace_current[link.neighbor_face]
                             : nb.trace_previous[link.neighbor_face];
      in.kind = FaceInflow::Kind::NEIGHBOR;
      in.trace = buffer.template data_ptr<double>();
      in.source_face_dofs = buffer.size(2);
      in.trace_map = link.trace_map.template data_ptr<int64_t>();
    } else if (link.type == physics::BoundaryType::REFLECTIVE) {
      // Reflected directions belong to other angle sets: lag them
      in.kind = FaceInflow::Kind::REFLECTIVE;
      in.trace = P.trace_previous[f].template data_ptr<double>();
      in.source_face_dofs = P.trace_previous[f].size(2);
      in.reflected = link.reflected.template data_ptr<int64_t>();
      in.albedo = link.albedo;
    }
    return in;
  }

  /// @brief One sweep task: sweep the units of chunk `c` of angle set
  /// `set_index` for group set `group_set`, in upwind order.
  void run_chunk(size_t set_index, size_t group_set, size_t c)
  {
    thread_local SweepWorkspace workspace;
    thread_local std::vector<double> moments;

    const auto& set = graph_->angle_sets()[set_index];
    const auto& dirs = set_directions_[set_index];
    const auto& groups = group_indices_[group_set];
    const int64_t M = harmonics_.size(1);
    const int64_t ng = static_cast<int64_t>(groups.size());

    SweepBlock block;
    block.directions = dirs.data();
    block.num_block_directions = static_cast<int64_t>(dirs.size());
    block.groups = groups.data();
    block.num_block_groups = ng;
    block.num_directions = omega_.size(0);
    block.num_groups = num_groups_;
    block.num_moments = M;
    block.omega = omega_.template data_ptr<double>();
    block.harmonics = harmonics_.template data_ptr<double>();
    block.weighted_harmonics = weighted_harmonics_.template data_ptr<double>();
    block.solver = config_.local_solver;

    for (int64_t local : schedules_[set_index][c].units) {
      auto& P = patches_[local];
      const int64_t N = P.disc->num_dofs();

      UnitSweep unit;
      unit.disc = P.disc.get();
      unit.sigma_t = P.sigma_t.template data_ptr<double>();
      unit.source_moments = P.source_moments.template data_ptr<double>();
      for (size_t f = 0; f < P.links.size(); f++) {
        unit.inflow[f] = face_inflow(set, local, f);
        unit.trace_out[f] = P.links[f].has_buffer
                              ? P.trace_current[f].template data_ptr<double>()
                              : nullptr;
      }
      unit.psi = P.psi.defined() ? P.psi.template data_ptr<double>() : nullptr;
      if (P.factors.defined()) {
        unit.factors = P.factors.template data_ptr<double>();
        unit.factor_pivots = P.factor_pivots.template data_ptr<int64_t>();
        unit.factors_ready = P.factors_ready;
      }
      moments.assign(static_cast<size_t>(M * ng * N), 0.0);
      unit.moments = moments.data();

      sweep_unit(unit, block, workspace);

      // Reduce this task's moment contribution (angle sets run concurrently)
      std::lock_guard<std::mutex> lock(*P.mutex);
      double* dst = P.moments_next.template data_ptr<double>();
      for (int64_t m = 0; m < M; m++) {
        for (int64_t bg = 0; bg < ng; bg++) {
          const double* src = moments.data() + (m * ng + bg) * N;
          double* out = dst + (m * num_groups_ + groups[bg]) * N;
          for (int64_t v = 0; v < N; v++)
            out[v] += src[v];
        }
      }
    }
  }

  /// @brief Scattering + external source moments from the current moments.
  void update_source_moments()
  {
    thread_local std::vector<double> scratch;
    const int64_t M = harmonics_.size(1);
    for (auto& P : patches_) {
      scattering_source(P.disc->view(),
        P.scatter.defined() ? P.scatter.template data_ptr<double>() : nullptr,
        P.scatter.defined() ? P.scatter.size(0) : 0, moment_order_raw_.data(),
        M, num_groups_, P.moments.template data_ptr<double>(),
        P.external.defined() ? P.external.template data_ptr<double>() : nullptr,
        P.source_moments.template data_ptr<double>(), scratch);
    }
  }

  /// @brief Build the sweep DAG: one task per (angle set, group set, chunk).
  void build_dag()
  {
    dag_.clear();
    schedules_.clear();
    const auto& sets = graph_->angle_sets();
    for (size_t a = 0; a < sets.size(); a++) {
      schedules_.push_back(graph_->schedule(a, chunk_of_));
      const auto& chunks = schedules_.back();
      for (size_t s = 0; s < group_sets_.size(); s++) {
        std::vector<task::Task*> tasks(chunks.size());
        for (size_t c = 0; c < chunks.size(); c++) {
          tasks[c] = dag_.create_task(task::DeviceTarget::CPU_ASYNC,
            "sweep_a" + std::to_string(a) + "_g" + std::to_string(s) + "_c" +
              std::to_string(c));
          tasks[c]->set_payload([this, a, s, c]() {
            auto t0 = std::chrono::steady_clock::now();
            run_chunk(a, s, c);
            task_nanoseconds_ +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();
            return task::TaskStatus::COMPLETED;
          });
        }
        for (size_t c = 0; c < chunks.size(); c++) {
          for (int64_t u : chunks[c].upstream)
            tasks[c]->add_dependency(tasks[u]);
        }
      }
    }
  }

public:
  // =================================================================
  // Public constructors
  /// @param mesh The (finalized or finalizable) mesh; every block local to
  /// this rank is a sweep unit.
  /// @param xs_server Multigroup cross sections.
  /// @param mpi_context The MPI context (same argument as TransportDriver).
  SweepDriver(typename Mesh::Ptr mesh, xs::Server::Ptr xs_server,
    const parallel::ParallelContext& mpi_context)
    : mesh_(std::move(mesh)), xs_server_(std::move(xs_server)),
      comm_(parallel::Communicator::world())
  {
    (void)mpi_context;
    if (!mesh_->is_finalized())
      mesh_->finalize();
    if (!xs_server_->is_finalized())
      xs_server_->finalize();
  }

  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new SweepDriver(std::forward<Args>(args)...));
  }

  // =================================================================
  // Public methods
  /// @brief Build the patch discretizations, sweep graph and task DAG.
  /// @param qset Angular quadrature (the one the TT solver uses).
  /// @param config Sweep configuration.
  /// @param chunks Optional GID -> chunk id map grouping units into tasks
  /// (e.g. FDG elements by parent patch). Empty uses
  /// config.task_granularity; otherwise every local GID must appear.
  void setup(const math::QuadratureSet::Ptr& qset, const SweepConfig& config,
    const std::unordered_map<int64_t, int64_t>& chunks = {})
  {
    qset_ = qset;
    config_ = config;
    has_fission_ = false;
    patches_.clear();
    gid_to_local_.clear();

    if (comm_.size() > 1) {
      throw utils::runtime_error(ctx("setup"),
        "Multi-rank sweeps are not implemented yet; run on one MPI rank");
    }

    std::tie(omega_, omega3_, weights_) = discrete_ordinates(*qset_, NumDim);
    const int64_t num_directions = omega_.size(0);

    // Patches
    int64_t num_orders = 1;
    for (const auto& block : mesh_->get_blocks()) {
      PatchState P;
      P.gid = block->get_gid();
      P.block = block;
      P.disc = PatchDiscretization::create(*block);
      const auto& mat = xs_server_->get_material(block->get_fill_id());
      P.sigma_t = mat.get_total().to(torch::kFloat64).cpu().contiguous();
      if (mat.get_scatter_gtg().defined() && mat.get_scatter_gtg().numel()) {
        P.scatter =
          mat.get_scatter_gtg().to(torch::kFloat64).cpu().contiguous();
        num_orders = std::max<int64_t>(num_orders, P.scatter.size(0));
      }
      has_fission_ = has_fission_ || mat.is_fissile();
      num_groups_ = P.sigma_t.size(0);

      const auto& fs = block->get_fixed_source();
      if (fs.has_value() && fs->defined()) {
        const int64_t N = P.disc->num_dofs();
        if (fs->function.has_value()) {
          throw utils::runtime_error(ctx("setup"),
            "Function fixed sources are not supported by the sweeper yet");
        }
        auto q = torch::zeros({num_groups_, N}, torch::kFloat64);
        if (fs->isotropic_strength.has_value()) {
          q += fs->isotropic_strength->to(torch::kFloat64).cpu().view({-1, 1}) *
               P.disc->basis_integral().view({1, -1});
        }
        if (fs->projected_isotropic_source.has_value()) {
          q += fs->projected_isotropic_source->to(torch::kFloat64)
                 .cpu()
                 .reshape({N, num_groups_})
                 .transpose(0, 1);
        }
        P.external = q.contiguous();
      }
      gid_to_local_[P.gid] = patches_.size();
      patches_.push_back(std::move(P));
    }

    std::tie(harmonics_, moment_order_) =
      discrete_harmonics(*qset_, NumDim, num_orders);
    weighted_harmonics_ = (harmonics_ * weights_.unsqueeze(1)).contiguous();
    moment_order_raw_.assign(moment_order_.template data_ptr<int64_t>(),
      moment_order_.template data_ptr<int64_t>() + moment_order_.numel());
    harmonics_ = harmonics_.contiguous();
    omega_ = omega_.contiguous();

    // Face links and interfaces
    std::vector<SweepInterface> interfaces;
    std::map<std::pair<int64_t, int64_t>, int64_t> owner_face_to_interface;
    for (size_t i = 0; i < patches_.size(); i++) {
      auto& P = patches_[i];
      P.links.resize(2 * NumDim);
      for (int64_t dim = 0; dim < NumDim; dim++) {
        for (bool is_upper : {false, true}) {
          const int64_t f = 2 * dim + (is_upper ? 1 : 0);
          const auto& info = P.block->get_boundary_info(dim, is_upper);
          auto& link = P.links[f];
          link.type = info.get_type();
          if (link.type == physics::BoundaryType::INCIDENT &&
              info.get_source().has_value() && info.get_source()->defined()) {
            throw utils::runtime_error(ctx("setup"),
              "Incident-flux boundary sources are not supported by the "
              "sweeper yet");
          }
          if (link.type == physics::BoundaryType::REFLECTIVE) {
            link.reflected = build_reflection(P.disc->faces()[f]).contiguous();
            link.albedo = info.albedo();
            link.has_buffer = true;
          } else if (link.type == physics::BoundaryType::INTERNAL) {
            const auto& conns = info.get_connections();
            if (conns.size() != 1) {
              throw utils::runtime_error(ctx("setup"),
                "Each internal face must connect to exactly one neighbor "
                "face (conforming interfaces only)");
            }
            const auto& conn = conns[0];
            auto it = gid_to_local_.find(conn.gid);
            if (it == gid_to_local_.end()) {
              throw utils::runtime_error(
                ctx("setup"), "Neighbor patch " + std::to_string(conn.gid) +
                                " is not on this rank");
            }
            link.neighbor = static_cast<int64_t>(it->second);
            link.neighbor_face = 2 * conn.dim + (conn.is_upper ? 1 : 0);
            link.trace_map =
              build_trace_map(dim, conn, *patches_[link.neighbor].disc)
                .contiguous();
            link.has_buffer = true;
            link.is_first = P.gid < conn.gid;
            if (link.is_first) {
              const auto& face = P.disc->faces()[f];
              SweepInterface itf;
              itf.first = static_cast<int64_t>(i);
              itf.second = link.neighbor;
              itf.forward_weight = face.upwind_weights(omega_, true).sum(1);
              itf.backward_weight = face.upwind_weights(omega_, false).sum(1);
              itf.tolerance = config_.face_weight_tol * face.area;
              link.interface = static_cast<int64_t>(interfaces.size());
              owner_face_to_interface[{static_cast<int64_t>(i), f}] =
                link.interface;
              interfaces.push_back(std::move(itf));
            }
          }
        }
      }
    }
    for (auto& P : patches_) {
      for (auto& link : P.links) {
        if (link.type == physics::BoundaryType::INTERNAL && !link.is_first) {
          link.interface =
            owner_face_to_interface.at({link.neighbor, link.neighbor_face});
        }
      }
    }
    graph_ = SweepGraph::create(static_cast<int64_t>(patches_.size()),
      std::move(interfaces), num_directions);

    // Chunks of units swept by one task
    chunk_of_.assign(patches_.size(), 0);
    for (size_t i = 0; i < patches_.size(); i++) {
      if (!chunks.empty()) {
        auto it = chunks.find(patches_[i].gid);
        if (it == chunks.end()) {
          throw utils::runtime_error(ctx("setup"),
            "`chunks` has no entry for GID " + std::to_string(patches_[i].gid));
        }
        chunk_of_[i] = it->second;
      } else if (config_.task_granularity == TaskGranularity::UNIT) {
        chunk_of_[i] = static_cast<int64_t>(i);
      }
    }

    // Plain index vectors for the kernel
    set_directions_.clear();
    for (const auto& set : graph_->angle_sets()) {
      const auto& d = set.directions;
      set_directions_.emplace_back(d.template data_ptr<int64_t>(),
        d.template data_ptr<int64_t>() + d.numel());
    }

    // Group sets
    group_sets_.clear();
    const int64_t gs = config_.group_set_size > 0
                         ? std::min(config_.group_set_size, num_groups_)
                         : num_groups_;
    for (int64_t g0 = 0; g0 < num_groups_; g0 += gs) {
      group_sets_.push_back(
        torch::arange(g0, std::min(g0 + gs, num_groups_), torch::kInt64));
    }
    group_indices_.clear();
    for (const auto& g : group_sets_) {
      group_indices_.emplace_back(g.template data_ptr<int64_t>(),
        g.template data_ptr<int64_t>() + g.numel());
    }

    // Buffers
    const int64_t M = harmonics_.size(1);
    for (auto& P : patches_) {
      const int64_t N = P.disc->num_dofs();
      P.trace_current.assign(P.links.size(), torch::Tensor());
      P.trace_previous.assign(P.links.size(), torch::Tensor());
      for (size_t f = 0; f < P.links.size(); f++) {
        if (P.links[f].has_buffer) {
          const int64_t Nf = P.disc->faces()[f].num_dofs();
          P.trace_current[f] =
            torch::zeros({num_directions, num_groups_, Nf}, torch::kFloat64);
          P.trace_previous[f] = torch::zeros_like(P.trace_current[f]);
        }
      }
      P.moments = torch::zeros({M, num_groups_, N}, torch::kFloat64);
      P.moments_next = torch::zeros_like(P.moments);
      P.source_moments = torch::zeros_like(P.moments);
      if (config_.cache_factorizations) {
        const bool banded = config_.local_solver == LocalSolverType::BANDED_LU;
        const int64_t size = banded ? N * P.disc->ldab() : N * N;
        P.factors =
          torch::empty({num_directions, num_groups_, size}, torch::kFloat64);
        P.factor_pivots =
          torch::empty({num_directions, num_groups_, N}, torch::kInt64);
      } else {
        P.factors = torch::Tensor();
        P.factor_pivots = torch::Tensor();
      }
      P.factors_ready = false;
      P.psi =
        config_.store_angular_flux
          ? torch::zeros({num_directions, num_groups_, N}, torch::kFloat64)
          : torch::Tensor();
    }

    scheduler_ = std::make_unique<task::TaskScheduler>(
      std::max<int64_t>(1, config_.num_threads));
    build_dag();
    is_setup_ = true;
  }

  /// @brief Run one transport sweep with the current flux moments as the
  /// scattering source, then advance the moments and face traces.
  void sweep()
  {
    if (!is_setup_)
      throw utils::runtime_error(ctx("sweep"), "Call setup() first");
    update_source_moments();
    for (auto& P : patches_) {
      double* m = P.moments_next.template data_ptr<double>();
      std::fill(m, m + P.moments_next.numel(), 0.0);
    }
    dag_.reset();
    scheduler_->execute(dag_);
    for (auto& P : patches_) {
      std::swap(P.trace_current, P.trace_previous);
      std::swap(P.moments, P.moments_next);
      // Every (direction, group) of every unit is solved once per sweep
      P.factors_ready = P.factors.defined();
    }
  }

  /// @brief Solve a fixed-source problem with unaccelerated source iteration.
  /// @return The solution: the angular flux if `store_angular_flux`, else
  /// the scalar flux.
  typename Solution::Ptr solve_fixed_source()
  {
    if (!is_setup_)
      throw utils::runtime_error(
        ctx("solve_fixed_source"), "Call setup() first");
    if (has_fission_)
      throw utils::runtime_error(ctx("solve_fixed_source"),
        "Fissile materials need the eigenvalue solver, which is not "
        "implemented yet");

    // Restart from zero flux
    for (auto& P : patches_) {
      P.moments.zero_();
      for (size_t f = 0; f < P.links.size(); f++) {
        if (P.links[f].has_buffer) {
          P.trace_current[f].zero_();
          P.trace_previous[f].zero_();
        }
      }
    }
    errors_.clear();
    sweep_times_.clear();
    converged_ = false;

    // Avoid oversubscription when tasks run concurrently
    const int previous_threads = at::get_num_threads();
    if (config_.num_threads > 1)
      at::set_num_threads(1);

    const bool verbose = config_.verbose && comm_.rank() == 0;
    auto start = std::chrono::high_resolution_clock::now();
    double previous_error = -1.0;
    int64_t it = 0;
    for (; it < config_.max_iter; it++) {
      // Scalar flux before the sweep (moment 0 is the leading G * N block)
      std::vector<std::vector<double>> old(patches_.size());
      for (size_t i = 0; i < patches_.size(); i++) {
        const double* phi = patches_[i].moments.template data_ptr<double>();
        old[i].assign(phi, phi + num_groups_ * patches_[i].disc->num_dofs());
      }

      auto t0 = std::chrono::high_resolution_clock::now();
      sweep();
      double sweep_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0)
                         .count();
      sweep_times_.push_back(sweep_s);

      double num = 0.0, den = 0.0;
      for (size_t i = 0; i < patches_.size(); i++) {
        const double* phi = patches_[i].moments.template data_ptr<double>();
        for (size_t k = 0; k < old[i].size(); k++) {
          const double diff = phi[k] - old[i][k];
          num += diff * diff;
          den += phi[k] * phi[k];
        }
      }
      double error = den > 0 ? std::sqrt(num / den) : 0.0;
      errors_.push_back(error);
      double rho = previous_error > 0 ? error / previous_error : 0.0;
      previous_error = error;
      double target = config_.tol;
      if (config_.rho_safety)
        target *= (1.0 - std::min(rho, 0.999));

      if (verbose) {
        std::cout << "-- sweep (" << it
                  << "): scalar flux error = " << std::scientific
                  << std::setprecision(4) << error << ", rho = " << std::fixed
                  << std::setprecision(4) << rho
                  << ", sweep = " << std::setprecision(3) << sweep_s
                  << " s, elapsed = "
                  << std::chrono::duration<double>(
                       std::chrono::high_resolution_clock::now() - start)
                       .count()
                  << " s" << std::defaultfloat << std::endl;
      }
      if (it > 0 && error < target) {
        converged_ = true;
        it++;
        break;
      }
    }
    num_iterations_ = std::min<int64_t>(it, config_.max_iter);
    if (config_.num_threads > 1)
      at::set_num_threads(previous_threads);
    if (verbose) {
      std::cout << "-- " << (converged_ ? "Converged!" : "Failed to Converge!")
                << " (" << num_iterations_ << " sweeps)" << std::endl;
    }
    return build_solution();
  }

  /// @brief Wrap the current flux in a TransportSolution so the existing
  /// error and comparison tooling applies.
  typename Solution::Ptr build_solution() const
  {
    const bool angular = config_.store_angular_flux;
    auto solution = Solution::create(parallel::Communicator::world(), mesh_,
      qset_, xs_server_, physics::DGTransportAssemblerConfig(), std::nullopt,
      angular);
    solution->set_iteration_counts(
      static_cast<int>(num_iterations_), static_cast<int>(num_iterations_));
    for (const auto& P : patches_) {
      auto dense = angular ? get_angular_flux(P.gid) : get_scalar_flux(P.gid);
      solution->add_local_patch(
        P.gid, linalg::State(linalg::TTEngine::from_dense(dense, 1e-14)));
    }
    return solution;
  }

  /// @return The scalar flux of a patch, shape (*ctrlpts_grid, G).
  torch::Tensor get_scalar_flux(int64_t gid) const
  {
    const auto& P = patches_.at(gid_to_local_.at(gid));
    auto shape = P.disc->shape();
    shape.push_back(num_groups_);
    return P.moments[0].transpose(0, 1).reshape(shape).contiguous();
  }

  /// @return The flux moments of a patch, shape (num_moments, G, N).
  const torch::Tensor& get_moments(int64_t gid) const
  {
    return patches_.at(gid_to_local_.at(gid)).moments;
  }

  /// @return The angular flux of a patch in the TT core order,
  /// (*angular_modes, *ctrlpts_grid, G).
  torch::Tensor get_angular_flux(int64_t gid) const
  {
    const auto& P = patches_.at(gid_to_local_.at(gid));
    if (!P.psi.defined()) {
      throw utils::runtime_error(ctx("get_angular_flux"),
        "The angular flux is only kept with store_angular_flux = true");
    }
    std::vector<int64_t> shape = angular_modes();
    for (auto s : P.disc->shape())
      shape.push_back(s);
    shape.push_back(num_groups_);
    return P.psi.permute({0, 2, 1}).reshape(shape).contiguous();
  }

  /// @return The factored angular mode sizes (TT core order).
  std::vector<int64_t> angular_modes() const
  {
    if (NumDim == 1)
      return {omega_.size(0)};
    const auto& pq = static_cast<const math::ProductQuadrature&>(*qset_);
    return {
      pq.get_quads()[0]->get_num_dofs(), pq.get_quads()[1]->get_num_dofs()};
  }

  /// @brief Dense matrix of a patch's full local operator with reflective
  /// faces implicit and internal faces excluded -- the operator the TT
  /// assembler stores as the LinearSystem's interior operator -- in the TT
  /// ordering (direction, dofs, group) for rows and columns. For
  /// verification on small problems only.
  /// @return (N_Omega * N * G, N_Omega * N * G) dense matrix.
  torch::Tensor assemble_dense_operator(int64_t gid) const
  {
    const auto& P = patches_.at(gid_to_local_.at(gid));
    const auto& disc = *P.disc;
    const int64_t D = omega_.size(0);
    const int64_t N = disc.num_dofs();
    const int64_t G = num_groups_;
    auto A = torch::zeros({D, N, G, D, N, G}, torch::kFloat64);

    // Streaming, collision and outflow (block diagonal in angle and group)
    auto values = disc.streaming_values(omega_);
    for (int64_t g = 0; g < G; g++) {
      auto L = disc.to_dense(
        values + P.sigma_t[g].template item<double>() * disc.mass_values());
      for (int64_t d = 0; d < D; d++) {
        A.index_put_(
          {d, torch::indexing::Slice(), g, d, torch::indexing::Slice(), g},
          L[d]);
      }
    }

    // Scattering: K(d, d') w_d' sigma_l(g, g') M
    auto M = disc.to_dense(disc.mass_values().unsqueeze(0))[0];
    const int64_t L = P.scatter.defined() ? P.scatter.size(0) : 0;
    for (int64_t m = 0; m < harmonics_.size(1); m++) {
      int64_t l = moment_order_[m].template item<int64_t>();
      if (l >= L)
        continue;
      auto K = torch::outer(
        harmonics_.select(1, m), weighted_harmonics_.select(1, m)); // (D, D')
      A -= torch::einsum("ab,vu,gh->avgbuh", {K, M, P.scatter[l]});
    }

    // Reflective inflow: rows d, columns reflected d*
    for (size_t f = 0; f < P.links.size(); f++) {
      const auto& link = P.links[f];
      if (link.type != physics::BoundaryType::REFLECTIVE)
        continue;
      const auto& face = disc.faces()[f];
      auto c = link.albedo * face.upwind_weights(
                               omega_.index_select(0, link.reflected), true);
      for (int64_t d = 0; d < D; d++) {
        int64_t ds = link.reflected[d].template item<int64_t>();
        auto B = torch::zeros({N, N}, torch::kFloat64);
        auto vals = c[d].index({face.pair_q}) * face.pair_val;
        B.index_put_(
          {face.dofs.index({face.pair_a}), face.dofs.index({face.pair_b})},
          vals, /*accumulate=*/true);
        for (int64_t g = 0; g < G; g++) {
          A.index(
             {d, torch::indexing::Slice(), g, ds, torch::indexing::Slice(), g})
            .sub_(B);
        }
      }
    }
    return A.reshape({D * N * G, D * N * G});
  }

  // =================================================================
  // Public getters
  const SweepConfig& get_config() const noexcept { return config_; }
  const SweepGraph::Ptr& get_graph() const noexcept { return graph_; }
  PatchDiscretization::Ptr get_discretization(int64_t gid) const
  {
    return patches_.at(gid_to_local_.at(gid)).disc;
  }
  const torch::Tensor& get_harmonics() const noexcept { return harmonics_; }
  const torch::Tensor& get_ordinates() const noexcept { return omega_; }
  const torch::Tensor& get_weights() const noexcept { return weights_; }
  const std::vector<double>& get_errors() const noexcept { return errors_; }
  const std::vector<double>& get_sweep_times() const noexcept
  {
    return sweep_times_;
  }
  int64_t get_num_iterations() const noexcept { return num_iterations_; }
  bool is_converged() const noexcept { return converged_; }
  /// @return Cumulative seconds spent inside sweep task bodies, summed over
  /// worker threads (compare with sweep wall time to see scheduler cost).
  double get_task_seconds() const noexcept
  {
    return static_cast<double>(task_nanoseconds_.load()) * 1e-9;
  }
  void reset_task_seconds() noexcept { task_nanoseconds_ = 0; }
  /// @return Bytes held by the factorization cache (0 if disabled).
  int64_t factorization_bytes() const noexcept
  {
    int64_t bytes = 0;
    for (const auto& P : patches_) {
      if (P.factors.defined())
        bytes += P.factors.numel() * 8 + P.factor_pivots.numel() * 8;
    }
    return bytes;
  }
  /// @return Chunk id of each local unit (in local order).
  const std::vector<int64_t>& get_chunks() const noexcept { return chunk_of_; }
  int64_t num_tasks() const noexcept
  {
    return static_cast<int64_t>(dag_.size());
  }
  const typename Mesh::Ptr& get_mesh() const noexcept { return mesh_; }
};

} // namespace ttnte::sweep
