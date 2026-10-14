#pragma once

#include "ttnte/driver/transport_solution.hpp"
#include "ttnte/linalg/krylov.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/load_balancer.hpp"
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
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
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
/// few coarse tasks rather than one per element.
///
/// Iterations follow the usual nested structure (as in OpenSn): an outer
/// loop (across-group-set passes for fixed source, power iteration for
/// k-eigenvalue) around a Gauss-Seidel or Jacobi pass over the group sets,
/// each group set solved for its within-set scattering by Richardson
/// iteration or matrix-free GMRES (one group-set sweep per product). GMRES
/// carries the lagged face traces (reflective faces and cycle-broken
/// interfaces) in its unknown, so the swept operator is exactly linear.
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
  /// Called after every `frequency`-th outer iteration of a solve.
  using Callback = std::function<void(const SweepDriver&)>;

private:
  // =================================================================
  // Private types
  /// Where a face's incoming angular flux comes from.
  struct FaceLink {
    physics::BoundaryType type = physics::BoundaryType::VACUUM;
    /// INTERNAL: neighbor local index (-1 if remote) and its face
    /// (2 * dim + is_upper).
    int64_t neighbor = -1;
    int64_t neighbor_face = -1;
    /// INTERNAL: MPI rank of the neighbor if it is on another rank (-1 if
    /// local; its traces then arrive in this patch's ghost buffers), and the
    /// number of dofs of its face.
    int remote_rank = -1;
    int64_t neighbor_face_dofs = 0;
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
    /// (1, G, G) fission transfer chi (x) nu_sigma_f (undefined if
    /// non-fissile), and (G, N) fission source of the running power
    /// iteration.
    torch::Tensor fission;
    torch::Tensor fission_source;
    /// (G, N) external source load vector (undefined if none).
    torch::Tensor external;
    std::vector<FaceLink> links;
    /// Per face (N_Omega, G, Nf) outgoing traces of this and the previous
    /// sweep (undefined if the face has no buffer).
    std::vector<torch::Tensor> trace_current;
    std::vector<torch::Tensor> trace_previous;
    /// Per face with a remote neighbor: (N_Omega, G, Nf_neighbor) neighbor
    /// traces of this and the previous sweep, received by MPI (undefined
    /// otherwise).
    std::vector<torch::Tensor> ghost_current;
    std::vector<torch::Tensor> ghost_previous;
    /// (num_moments, G, N) moments, the moments being accumulated by the
    /// running sweep, and the scattering + external source moments.
    torch::Tensor moments;
    torch::Tensor moments_next;
    torch::Tensor source_moments;
    /// (num_moments, G, N) GMRES iterate of the within-group-set solve, and
    /// the moments at the start of the pass (Jacobi coupling).
    torch::Tensor krylov_moments;
    torch::Tensor pass_moments;
    /// (N_Omega, G, N) angular flux, if stored.
    torch::Tensor psi;
    /// Factorization cache (see SweepConfig::cache_factorizations).
    torch::Tensor factors;
    torch::Tensor factor_pivots;
    /// Per group set: whether the cached factors are computed.
    std::vector<uint8_t> factors_ready;
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

  /// Lagged face traces read from the previous sweep: (unit, face,
  /// directions). They are part of the GMRES unknown.
  struct DelayedBlock {
    int64_t local = -1;
    int64_t face = -1;
    std::vector<int64_t> directions;
    /// The lagged trace is a ghost (remote upstream) of `local`'s face
    /// rather than `local`'s own outgoing trace.
    bool ghost = false;
  };
  std::vector<DelayedBlock> delayed_;
  /// Per group set: (G) membership mask and its complement.
  std::vector<std::vector<uint8_t>> group_masks_;
  std::vector<std::vector<uint8_t>> group_complements_;

  /// Directions swept by one task: a block of one angle set's directions
  /// (SweepConfig::directions_per_task).
  struct DirectionBlock {
    size_t set = 0;
    std::vector<int64_t> directions;
  };
  std::vector<DirectionBlock> direction_blocks_;

  /// MPI trace exchange of one cross-rank interface for an angle set: the
  /// local unit's face and the remote rank (see build_dag()).
  struct Exchange {
    int64_t local = -1;
    int64_t face = -1;
    int rank = -1;
    /// Index of the interface among those shared with `rank` (message tag).
    int64_t pair_index = -1;
    /// Receives only: whether the local unit's sweep waits for it (else
    /// the trace is lagged and read next sweep).
    bool dependency = false;
  };
  /// Per angle set: traces to receive into ghosts and to send.
  std::vector<std::vector<Exchange>> recvs_;
  std::vector<std::vector<Exchange>> sends_;
  /// Per angle set: task (schedule chunk) of every local unit.
  std::vector<std::vector<int64_t>> task_of_unit_;
  /// Message buffers of the exchange tasks (stable addresses).
  std::deque<std::vector<double>> exchange_buffers_;
  /// Communicator of the trace exchange (a duplicate of comm_).
  parallel::Communicator sweep_comm_;
  /// Whether this driver solves the whole mesh on this rank alone.
  bool serial_ = false;
  /// Optional user partition (GID -> rank); METIS when empty.
  std::unordered_map<int64_t, int> partition_;
  /// Rank of every node of the global sweep graph, and the local index of
  /// each node (-1 if remote) and node of each local unit.
  std::vector<int> node_rank_;
  std::vector<int64_t> node_to_local_;
  std::vector<int64_t> local_to_node_;

  /// One sweep DAG per group set.
  std::vector<std::unique_ptr<task::TaskGraph>> dags_;
  std::unique_ptr<task::TaskScheduler> scheduler_;
  /// Flat DAG of index-range tasks running range_fn_ (see parallel_for()).
  std::unique_ptr<task::TaskGraph> range_dag_;
  const std::function<void(int64_t)>* range_fn_ = nullptr;
  int64_t range_size_ = 0;
  bool is_setup_ = false;
  /// Whether any local material is fissile.
  bool has_fission_ = false;
  /// Cumulative time spent inside sweep task bodies (all threads).
  std::atomic<int64_t> task_nanoseconds_ {0};
  /// Wall time spent executing sweep DAGs (the parallel part of a sweep).
  int64_t dag_nanoseconds_ = 0;

  /// Optional per-outer-iteration callback and its frequency.
  Callback callback_;
  int64_t callback_frequency_ = 1;
  /// Wall time of the running solve at the last completed outer iteration,
  /// excluding time spent in the callback.
  double solve_seconds_ = 0.0;

  /// Whether the running solve is a k-eigenvalue power iteration, and k.
  bool eigenvalue_mode_ = false;
  double k_ = 1.0;

  /// Convergence history of the last solve (per outer iteration).
  std::vector<double> errors_;
  std::vector<double> k_history_;
  std::vector<double> sweep_times_;
  std::vector<int64_t> outer_sweeps_;
  int64_t num_iterations_ = 0;
  bool converged_ = false;
  /// Group-set sweeps of the last solve, and the same in full-sweep
  /// equivalents (a group-set sweep counts |set| / G).
  int64_t group_set_sweeps_ = 0;
  double equivalent_sweeps_ = 0.0;

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

  /// @return The trace buffer of a lagged block (its current or previous
  /// sweep).
  torch::Tensor& delayed_buffer(const DelayedBlock& blk, bool current)
  {
    auto& P = patches_[blk.local];
    if (blk.ghost)
      return current ? P.ghost_current[blk.face] : P.ghost_previous[blk.face];
    return current ? P.trace_current[blk.face] : P.trace_previous[blk.face];
  }
  const torch::Tensor& delayed_buffer(
    const DelayedBlock& blk, bool current) const
  {
    return const_cast<SweepDriver*>(this)->delayed_buffer(blk, current);
  }

  /// @return The sum of `value` over all ranks.
  double global_sum(double value) const
  {
    if (comm_.size() == 1)
      return value;
    double sum = 0.0;
    comm_.iallreduce(&value, &sum, 1, parallel::MPIOp::SUM).wait();
    return sum;
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
      const bool current = state == EdgeState::DEPENDENCY;
      const auto& buffer =
        link.remote_rank >= 0
          ? (current ? P.ghost_current[f] : P.ghost_previous[f])
          : (current
                ? patches_[link.neighbor].trace_current[link.neighbor_face]
                : patches_[link.neighbor].trace_previous[link.neighbor_face]);
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
  void run_chunk(size_t block_index, size_t group_set, size_t c)
  {
    thread_local SweepWorkspace workspace;
    thread_local std::vector<double> moments;

    const auto& blk = direction_blocks_[block_index];
    const size_t set_index = blk.set;
    const auto& set = graph_->angle_sets()[set_index];
    const auto& dirs = blk.directions;
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
        unit.factors_ready = P.factors_ready[group_set] != 0;
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

  /// Which moments feed the within-group-set scattering term.
  enum class Within : uint8_t { NONE, MOMENTS, KRYLOV };

  /// @brief Source moments of group set `s` for every unit.
  /// @param within Moments of the within-set scattering term (NONE leaves it
  /// out, KRYLOV uses the GMRES iterate).
  /// @param fixed Include the parts that do not depend on this set's flux:
  /// external and fission sources, and scattering from the other sets.
  /// @param jacobi Take the other sets' flux from the start of the pass.
  /// @brief Call `fn(i)` for i in [0, n), in parallel on the scheduler's
  /// workers (`fn` must be safe for distinct i; it may use thread_local
  /// scratch). Serial with one thread.
  void parallel_for(int64_t n, const std::function<void(int64_t)>& fn)
  {
    if (!range_dag_ || n < 2) {
      for (int64_t i = 0; i < n; i++)
        fn(i);
      return;
    }
    range_fn_ = &fn;
    range_size_ = n;
    range_dag_->reset();
    scheduler_->execute(*range_dag_, task::ExecutionMode::PUSH);
    range_fn_ = nullptr;
  }

  /// @brief Apply `fn` to every patch in parallel (see parallel_for()).
  void for_each_patch(const std::function<void(PatchState&)>& fn)
  {
    parallel_for(static_cast<int64_t>(patches_.size()),
      [&](int64_t i) { fn(patches_[i]); });
  }

  /// @brief Build the range tasks of parallel_for(): task c of K handles
  /// [c n / K, (c + 1) n / K). None (serial) with one thread.
  void build_range_dag()
  {
    range_dag_.reset();
    const int64_t num_threads = std::max<int64_t>(1, config_.num_threads);
    if (num_threads == 1)
      return;
    const int64_t num_chunks = 4 * num_threads;
    range_dag_ = std::make_unique<task::TaskGraph>();
    for (int64_t c = 0; c < num_chunks; c++) {
      auto* task = range_dag_->create_task(
        task::DeviceTarget::CPU_ASYNC, "range_" + std::to_string(c));
      task->set_payload([this, c, num_chunks]() {
        const int64_t begin = c * range_size_ / num_chunks;
        const int64_t end = (c + 1) * range_size_ / num_chunks;
        for (int64_t i = begin; i < end; i++)
          (*range_fn_)(i);
        return task::TaskStatus::COMPLETED;
      });
    }
  }

  void compute_source(size_t s, Within within, bool fixed, bool jacobi)
  {
    thread_local std::vector<double> scratch;
    const auto& groups = group_indices_[s];
    const int64_t ng = static_cast<int64_t>(groups.size());
    const int64_t M = harmonics_.size(1);
    const int64_t G = num_groups_;
    for_each_patch([&](PatchState& P) {
      const auto& view = P.disc->view();
      const int64_t N = view.num_dofs;
      double* out = P.source_moments.template data_ptr<double>();
      for (int64_t m = 0; m < M; m++)
        for (int64_t g : groups)
          std::fill(out + (m * G + g) * N, out + (m * G + g + 1) * N, 0.0);

      const bool scatters = P.scatter.defined();
      const double* sigma_s =
        scatters ? P.scatter.template data_ptr<double>() : nullptr;
      const int64_t L = scatters ? P.scatter.size(0) : 0;
      if (fixed) {
        const double* q = nullptr;
        if (eigenvalue_mode_ && P.fission_source.defined())
          q = P.fission_source.template data_ptr<double>();
        else if (!eigenvalue_mode_ && P.external.defined())
          q = P.external.template data_ptr<double>();
        if (q) {
          for (int64_t g : groups)
            for (int64_t i = 0; i < N; i++)
              out[g * N + i] += q[g * N + i];
        }
        if (scatters && static_cast<int64_t>(group_sets_.size()) > 1) {
          const auto& other = jacobi ? P.pass_moments : P.moments;
          transfer_source(view, sigma_s, L, moment_order_raw_.data(), M, G,
            other.template data_ptr<double>(), 1.0, groups.data(), ng,
            group_complements_[s].data(), out, scratch);
        }
      }
      if (within != Within::NONE && scatters) {
        const auto& phi =
          within == Within::KRYLOV ? P.krylov_moments : P.moments;
        transfer_source(view, sigma_s, L, moment_order_raw_.data(), M, G,
          phi.template data_ptr<double>(), 1.0, groups.data(), ng,
          group_masks_[s].data(), out, scratch);
      }
    });
  }

  /// @brief Sweep group set `s` with the current source moments and lagged
  /// traces, accumulating its moments into moments_next.
  void sweep_group_set(size_t s)
  {
    const auto& groups = group_indices_[s];
    const int64_t M = harmonics_.size(1);
    for_each_patch([&](PatchState& P) {
      const int64_t N = P.disc->num_dofs();
      double* m = P.moments_next.template data_ptr<double>();
      for (int64_t k = 0; k < M; k++)
        for (int64_t g : groups)
          std::fill(m + (k * num_groups_ + g) * N,
            m + (k * num_groups_ + g + 1) * N, 0.0);
    });
    dags_[s]->reset();
    auto t0 = std::chrono::steady_clock::now();
    scheduler_->execute(*dags_[s], task::ExecutionMode::PUSH);
    dag_nanoseconds_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - t0)
                          .count();
    for (auto& P : patches_)
      if (P.factors.defined())
        P.factors_ready[s] = 1;
    group_set_sweeps_++;
    equivalent_sweeps_ +=
      static_cast<double>(groups.size()) / static_cast<double>(num_groups_);
  }

  /// @brief Accept the last sweep of group set `s`: moments and outgoing
  /// traces become the current iterate.
  void commit_group_set(size_t s)
  {
    const auto& groups = group_indices_[s];
    const int64_t M = harmonics_.size(1);
    const int64_t G = num_groups_;
    for_each_patch([&](PatchState& P) {
      const int64_t N = P.disc->num_dofs();
      const double* next = P.moments_next.template data_ptr<double>();
      double* cur = P.moments.template data_ptr<double>();
      for (int64_t m = 0; m < M; m++)
        for (int64_t g : groups)
          std::copy(next + (m * G + g) * N, next + (m * G + g + 1) * N,
            cur + (m * G + g) * N);
    });
    // Previous traces are read only where they are lagged (reflective faces
    // and LAGGED interface edges, the delayed blocks), so only those
    // directions are carried over
    parallel_for(static_cast<int64_t>(delayed_.size()), [&](int64_t b) {
      const auto& blk = delayed_[b];
      const auto& current = delayed_buffer(blk, true);
      const int64_t Nf = current.size(2);
      const double* src = current.template data_ptr<double>();
      double* dst = delayed_buffer(blk, false).template data_ptr<double>();
      for (int64_t d : blk.directions)
        for (int64_t g : groups)
          std::copy(src + (d * G + g) * Nf, src + (d * G + g + 1) * Nf,
            dst + (d * G + g) * Nf);
    });
  }

  /// @return Length of the GMRES unknown of group set `s`.
  int64_t krylov_size(size_t s) const
  {
    const int64_t ng = static_cast<int64_t>(group_indices_[s].size());
    const int64_t M = harmonics_.size(1);
    int64_t n = 0;
    for (const auto& P : patches_)
      n += M * ng * P.disc->num_dofs();
    for (const auto& blk : delayed_)
      n += static_cast<int64_t>(blk.directions.size()) * ng *
           delayed_buffer(blk, false).size(2);
    return n;
  }

  /// @brief Gather the moments of group set `s` (from moments or
  /// moments_next) and its lagged traces (from the previous or current
  /// traces) into a Krylov vector.
  void pack(size_t s, bool from_next, bool from_current, double* x) const
  {
    const auto& groups = group_indices_[s];
    const int64_t M = harmonics_.size(1);
    const int64_t G = num_groups_;
    for (const auto& P : patches_) {
      const int64_t N = P.disc->num_dofs();
      const auto& src = from_next ? P.moments_next : P.moments;
      const double* m = src.template data_ptr<double>();
      for (int64_t k = 0; k < M; k++)
        for (int64_t g : groups) {
          std::copy(m + (k * G + g) * N, m + (k * G + g + 1) * N, x);
          x += N;
        }
    }
    for (const auto& blk : delayed_) {
      const auto& buf = delayed_buffer(blk, from_current);
      const int64_t Nf = buf.size(2);
      const double* t = buf.template data_ptr<double>();
      for (int64_t d : blk.directions)
        for (int64_t g : groups) {
          std::copy(t + (d * G + g) * Nf, t + (d * G + g + 1) * Nf, x);
          x += Nf;
        }
    }
  }

  /// @brief Scatter a Krylov vector of group set `s` into the moments (the
  /// GMRES iterate or the current moments) and the lagged (previous)
  /// traces.
  void unpack(size_t s, const double* x, bool to_krylov)
  {
    const auto& groups = group_indices_[s];
    const int64_t M = harmonics_.size(1);
    const int64_t G = num_groups_;
    for (auto& P : patches_) {
      const int64_t N = P.disc->num_dofs();
      auto& dst = to_krylov ? P.krylov_moments : P.moments;
      double* m = dst.template data_ptr<double>();
      for (int64_t k = 0; k < M; k++)
        for (int64_t g : groups) {
          std::copy(x, x + N, m + (k * G + g) * N);
          x += N;
        }
    }
    for (const auto& blk : delayed_) {
      auto& buf = delayed_buffer(blk, false);
      const int64_t Nf = buf.size(2);
      double* t = buf.template data_ptr<double>();
      for (int64_t d : blk.directions)
        for (int64_t g : groups) {
          std::copy(x, x + Nf, t + (d * G + g) * Nf);
          x += Nf;
        }
    }
  }

  /// @return Within-group-set tolerance of the current outer iteration.
  double inner_tolerance() const
  {
    const double floor =
      config_.inner_tol > 0 ? config_.inner_tol : 0.1 * config_.tol;
    if (config_.inner_forcing <= 0)
      return floor;
    const double change = errors_.empty() ? 1.0 : std::min(errors_.back(), 1.0);
    return std::max(floor, config_.inner_forcing * change);
  }

  /// @brief Solve group set `s` for its within-set scattering.
  /// @return Group-set sweeps used.
  int64_t inner_solve(size_t s, bool jacobi)
  {
    const int64_t start = group_set_sweeps_;
    const double tol = inner_tolerance();

    if (config_.inner_solver == InnerSolver::RICHARDSON) {
      const auto& groups = group_indices_[s];
      for (int64_t it = 0; it < config_.inner_max_iter; it++) {
        double num = 0.0, den = 0.0;
        compute_source(s, Within::MOMENTS, true, jacobi);
        sweep_group_set(s);
        for (const auto& P : patches_) {
          const int64_t N = P.disc->num_dofs();
          const double* a = P.moments.template data_ptr<double>();
          const double* b = P.moments_next.template data_ptr<double>();
          for (int64_t g : groups)
            for (int64_t i = g * N; i < (g + 1) * N; i++) {
              num += (b[i] - a[i]) * (b[i] - a[i]);
              den += b[i] * b[i];
            }
        }
        commit_group_set(s);
        num = global_sum(num);
        den = global_sum(den);
        if (den == 0.0 || std::sqrt(num / den) < tol)
          break;
      }
      return group_set_sweeps_ - start;
    }

    // GMRES on (I - T) x = T(0 source...) with T the swept operator
    const int64_t n = krylov_size(s);
    std::vector<double> x(n), b(n), tmp(n);
    pack(s, false, false, x.data()); // warm start from the current iterate

    // b = sweep of the fixed sources with zero iterate and lagged traces
    std::fill(tmp.begin(), tmp.end(), 0.0);
    unpack(s, tmp.data(), true);
    compute_source(s, Within::NONE, true, jacobi);
    sweep_group_set(s);
    pack(s, true, true, b.data());

    auto apply = [&](const double* in, double* out) {
      unpack(s, in, true);
      compute_source(s, Within::KRYLOV, false, jacobi);
      sweep_group_set(s);
      pack(s, true, true, tmp.data());
      for (int64_t i = 0; i < n; i++)
        out[i] = in[i] - tmp[i];
    };
    auto result = linalg::krylov::gmres(apply, b, x, config_.gmres_restart, tol,
      config_.inner_max_iter,
      comm_.size() > 1
        ? linalg::krylov::Reducer([this](double v) { return global_sum(v); })
        : linalg::krylov::Reducer());
    if (!result.converged && config_.verbose && comm_.rank() == 0) {
      std::cout << "-- warning: within-group-set GMRES (set " << s
                << ") stopped at relative residual " << result.relative_residual
                << std::endl;
    }

    // Consistent final sweep from the solution (traces, angular flux)
    unpack(s, x.data(), false);
    compute_source(s, Within::MOMENTS, true, jacobi);
    sweep_group_set(s);
    commit_group_set(s);
    return group_set_sweeps_ - start;
  }

  /// @brief One across-group-set pass (Gauss-Seidel or Jacobi).
  /// @return Group-set sweeps used.
  int64_t group_set_pass()
  {
    const bool jacobi = config_.group_iteration == GroupIteration::JACOBI;
    if (jacobi)
      for_each_patch([](PatchState& P) { P.pass_moments.copy_(P.moments); });
    int64_t sweeps = 0;
    for (size_t s = 0; s < group_sets_.size(); s++)
      sweeps += inner_solve(s, jacobi);
    return sweeps;
  }

  /// @return The scalar flux (moment 0) of every unit, concatenated.
  std::vector<double> scalar_flux_snapshot() const
  {
    std::vector<double> phi;
    for (const auto& P : patches_) {
      const double* m = P.moments.template data_ptr<double>();
      phi.insert(phi.end(), m, m + num_groups_ * P.disc->num_dofs());
    }
    return phi;
  }

  /// @return ||phi - old|| / ||denominator|| over the scalar flux.
  double scalar_flux_change(
    const std::vector<double>& old, bool relative_to_old) const
  {
    const auto phi = scalar_flux_snapshot();
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < phi.size(); i++) {
      num += (phi[i] - old[i]) * (phi[i] - old[i]);
      den += relative_to_old ? old[i] * old[i] : phi[i] * phi[i];
    }
    num = global_sum(num);
    den = global_sum(den);
    return den > 0 ? std::sqrt(num / den) : 0.0;
  }

  /// @return Total fission production of the current flux.
  double fission_production() const
  {
    double total = 0.0;
    for (const auto& P : patches_) {
      if (!P.fission.defined())
        continue;
      total +=
        transfer_total(P.disc->view(), P.fission.template data_ptr<double>(),
          num_groups_, P.moments.template data_ptr<double>(),
          P.disc->basis_integral().template data_ptr<double>());
    }
    return global_sum(total);
  }

  /// @brief Reset the iterate: zero traces, and zero or unit scalar flux.
  void reset_state(bool unit_scalar_flux)
  {
    for (auto& P : patches_) {
      P.moments.zero_();
      if (unit_scalar_flux)
        P.moments[0].fill_(1.0);
      for (size_t f = 0; f < P.links.size(); f++) {
        if (P.links[f].has_buffer) {
          P.trace_current[f].zero_();
          P.trace_previous[f].zero_();
        }
        if (P.ghost_current[f].defined()) {
          P.ghost_current[f].zero_();
          P.ghost_previous[f].zero_();
        }
      }
    }
    errors_.clear();
    k_history_.clear();
    sweep_times_.clear();
    outer_sweeps_.clear();
    group_set_sweeps_ = 0;
    equivalent_sweeps_ = 0.0;
    converged_ = false;
  }

  /// @brief Assign every node to a rank (the user partition, else METIS on
  /// the patch connectivity graph) and cull the mesh to this rank's blocks.
  /// @return The rank of every node.
  std::vector<int> partition(const std::vector<PatchState>& nodes)
  {
    std::vector<int> rank(nodes.size(), comm_.rank());
    if (comm_.size() == 1)
      return rank;
    std::unordered_map<int64_t, int> gid2rank = partition_;
    if (gid2rank.empty()) {
      parallel::LoadBalancer<BlockType> balancer(comm_.size());
      gid2rank =
        balancer.compute_partition(mesh_->build_connectivity_graph(), comm_)
          .second;
    }
    std::vector<int64_t> keep;
    for (size_t i = 0; i < nodes.size(); i++) {
      auto it = gid2rank.find(nodes[i].gid);
      if (it == gid2rank.end() || it->second < 0 ||
          it->second >= comm_.size()) {
        throw utils::runtime_error(
          ctx("setup"), "The partition has no valid rank for GID " +
                          std::to_string(nodes[i].gid));
      }
      rank[i] = it->second;
      if (rank[i] == comm_.rank())
        keep.push_back(nodes[i].gid);
    }
    mesh_->cull_blocks(torch::tensor(keep, torch::kInt64), gid2rank);
    return rank;
  }

  /// @brief Build one sweep DAG per group set: one task per (direction
  /// block, chunk), plus MPI receive/send tasks for cross-rank interfaces.
  void build_dag()
  {
    dags_.clear();
    schedules_.clear();
    task_of_unit_.clear();
    recvs_.clear();
    sends_.clear();
    exchange_buffers_.clear();
    const auto& sets = graph_->angle_sets();
    const auto& interfaces = graph_->interfaces();
    const int64_t num_nodes = static_cast<int64_t>(node_rank_.size());

    // Schedule the global graph with every remote node in a chunk of its
    // own, then keep the local tasks: a dependency through a remote node is
    // carried by the MPI messages
    int64_t next_chunk = 0;
    for (int64_t c : chunk_of_)
      next_chunk = std::max(next_chunk, c + 1);
    std::vector<int64_t> chunk_of_node(num_nodes);
    for (int64_t node = 0; node < num_nodes; node++) {
      chunk_of_node[node] = node_to_local_[node] >= 0
                              ? chunk_of_[node_to_local_[node]]
                              : next_chunk + node;
    }
    for (size_t a = 0; a < sets.size(); a++) {
      auto global = graph_->schedule(a, chunk_of_node);
      std::vector<int64_t> local_task(global.size(), -1);
      std::vector<SweepChunk> chunks;
      for (size_t t = 0; t < global.size(); t++) {
        int64_t num_local = 0;
        for (int64_t node : global[t].units)
          num_local += node_to_local_[node] >= 0 ? 1 : 0;
        if (num_local == 0)
          continue;
        if (num_local != static_cast<int64_t>(global[t].units.size())) {
          throw utils::runtime_error(ctx("setup"),
            "Task chunking creates a cycle through another rank's patches; "
            "use TaskGranularity::UNIT for multi-rank sweeps");
        }
        local_task[t] = static_cast<int64_t>(chunks.size());
        SweepChunk chunk;
        for (int64_t node : global[t].units)
          chunk.units.push_back(node_to_local_[node]);
        chunks.push_back(std::move(chunk));
      }
      for (size_t t = 0; t < global.size(); t++) {
        if (local_task[t] < 0)
          continue;
        for (int64_t u : global[t].upstream)
          if (local_task[u] >= 0)
            chunks[local_task[t]].upstream.push_back(local_task[u]);
      }
      std::vector<int64_t> task_of(patches_.size(), -1);
      for (size_t c = 0; c < chunks.size(); c++)
        for (int64_t u : chunks[c].units)
          task_of[u] = static_cast<int64_t>(c);
      schedules_.push_back(std::move(chunks));
      task_of_unit_.push_back(std::move(task_of));
    }

    // Cross-rank interfaces: the local side's face and the interface's
    // index among those shared with the other rank (same on both ranks,
    // which enumerate the global interfaces in the same order)
    std::vector<int64_t> pair_index(interfaces.size(), -1);
    std::unordered_map<int, int64_t> pair_count;
    std::unordered_map<int64_t, std::pair<int64_t, int64_t>> local_side;
    for (size_t i = 0; i < patches_.size(); i++) {
      const auto& links = patches_[i].links;
      for (size_t f = 0; f < links.size(); f++) {
        if (links[f].remote_rank >= 0)
          local_side[links[f].interface] = {static_cast<int64_t>(i), f};
      }
    }
    for (size_t k = 0; k < interfaces.size(); k++) {
      auto it = local_side.find(static_cast<int64_t>(k));
      if (it == local_side.end())
        continue;
      const auto& P = patches_[it->second.first];
      pair_index[k] = pair_count[P.links[it->second.second].remote_rank]++;
    }
    recvs_.assign(sets.size(), {});
    sends_.assign(sets.size(), {});
    for (size_t a = 0; a < sets.size(); a++) {
      for (const auto& [k, side] : local_side) {
        const auto& link = patches_[side.first].links[side.second];
        const EdgeState in =
          link.is_first ? sets[a].backward[k] : sets[a].forward[k];
        const EdgeState out =
          link.is_first ? sets[a].forward[k] : sets[a].backward[k];
        Exchange ex {side.first, side.second, link.remote_rank, pair_index[k]};
        if (in != EdgeState::NONE) {
          ex.dependency = in == EdgeState::DEPENDENCY;
          recvs_[a].push_back(ex);
        }
        if (out != EdgeState::NONE) {
          ex.dependency = false;
          sends_[a].push_back(ex);
        }
      }
    }

    // Independent copies of an angle set's DAG over blocks of its
    // directions expose more concurrent tasks
    direction_blocks_.clear();
    for (size_t a = 0; a < sets.size(); a++) {
      const auto& dirs = set_directions_[a];
      const int64_t n = static_cast<int64_t>(dirs.size());
      const int64_t per =
        config_.directions_per_task > 0 ? config_.directions_per_task : n;
      for (int64_t d0 = 0; d0 < n; d0 += per) {
        direction_blocks_.push_back(
          {a, std::vector<int64_t>(
                dirs.begin() + d0, dirs.begin() + std::min(d0 + per, n))});
      }
    }

    for (size_t s = 0; s < group_sets_.size(); s++) {
      auto dag = std::make_unique<task::TaskGraph>();
      for (size_t b = 0; b < direction_blocks_.size(); b++) {
        const auto& chunks = schedules_[direction_blocks_[b].set];
        std::vector<task::Task*> tasks(chunks.size());
        for (size_t c = 0; c < chunks.size(); c++) {
          tasks[c] = dag->create_task(task::DeviceTarget::CPU_ASYNC,
            "sweep_b" + std::to_string(b) + "_g" + std::to_string(s) + "_c" +
              std::to_string(c));
          tasks[c]->set_payload([this, b, s, c]() {
            auto t0 = std::chrono::steady_clock::now();
            run_chunk(b, s, c);
            task_nanoseconds_ +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();
            return task::TaskStatus::COMPLETED;
          });
        }
        for (size_t c = 0; c < chunks.size(); c++)
          for (int64_t u : chunks[c].upstream)
            tasks[c]->add_dependency(tasks[u]);
        add_exchange_tasks(*dag, tasks, b, s);
      }
      dags_.push_back(std::move(dag));
    }
  }

  /// @brief MPI tasks of direction block `b` for group set `s`: receive the
  /// remote upstream traces into ghosts (the local sweep task waits for
  /// DEPENDENCY edges) and send the local outgoing traces after their task.
  /// Messages carry the block's directions x the set's groups x face dofs.
  void add_exchange_tasks(task::TaskGraph& dag,
    const std::vector<task::Task*>& tasks, size_t b, size_t s)
  {
    const size_t a = direction_blocks_[b].set;
    if (recvs_[a].empty() && sends_[a].empty())
      return;
    const int64_t num_blocks = static_cast<int64_t>(direction_blocks_.size());
    const int64_t nd =
      static_cast<int64_t>(direction_blocks_[b].directions.size());
    const int64_t ng = static_cast<int64_t>(group_indices_[s].size());
    const int64_t tag_bound = sweep_comm_.tag_upper_bound();

    auto make_tag = [&](const Exchange& ex) {
      const int64_t tag = ex.pair_index * num_blocks + static_cast<int64_t>(b);
      if (tag > tag_bound) {
        throw utils::runtime_error(ctx("setup"),
          "Too many cross-rank interfaces x direction blocks for MPI tags (" +
            std::to_string(tag) + " > MPI_TAG_UB " + std::to_string(tag_bound) +
            "); raise directions_per_task");
      }
      return static_cast<int>(tag);
    };

    for (const auto& ex : recvs_[a]) {
      const int64_t Nf = patches_[ex.local].links[ex.face].neighbor_face_dofs;
      exchange_buffers_.emplace_back(nd * ng * Nf);
      double* buf = exchange_buffers_.back().data();
      const int count = static_cast<int>(nd * ng * Nf);
      auto* recv = dag.create_task(task::DeviceTarget::NETWORK_ASYNC,
        "recv_b" + std::to_string(b) + "_u" + std::to_string(ex.local) + "_f" +
          std::to_string(ex.face));
      recv->set_payload(
        [this, buf, count, Nf, b, s, local = ex.local, face = ex.face,
          rank = ex.rank, tag = make_tag(ex), request = parallel::Request(),
          initiated = false]() mutable -> task::TaskStatus {
          if (!initiated) {
            request = sweep_comm_.irecv(buf, count, rank, tag);
            initiated = true;
            return task::TaskStatus::POLLING;
          }
          if (!request.test())
            return task::TaskStatus::POLLING;
          initiated = false;
          request = parallel::Request();
          double* ghost =
            patches_[local].ghost_current[face].template data_ptr<double>();
          const double* src = buf;
          for (int64_t d : direction_blocks_[b].directions)
            for (int64_t g : group_indices_[s]) {
              std::copy(src, src + Nf, ghost + (d * num_groups_ + g) * Nf);
              src += Nf;
            }
          return task::TaskStatus::COMPLETED;
        });
      if (ex.dependency)
        tasks[task_of_unit_[a][ex.local]]->add_dependency(recv);
    }

    for (const auto& ex : sends_[a]) {
      const int64_t Nf = patches_[ex.local].trace_current[ex.face].size(2);
      exchange_buffers_.emplace_back(nd * ng * Nf);
      double* buf = exchange_buffers_.back().data();
      const int count = static_cast<int>(nd * ng * Nf);
      auto* send = dag.create_task(task::DeviceTarget::NETWORK_ASYNC,
        "send_b" + std::to_string(b) + "_u" + std::to_string(ex.local) + "_f" +
          std::to_string(ex.face));
      send->set_payload(
        [this, buf, count, Nf, b, s, local = ex.local, face = ex.face,
          rank = ex.rank, tag = make_tag(ex), request = parallel::Request(),
          initiated = false]() mutable -> task::TaskStatus {
          if (!initiated) {
            const double* trace =
              patches_[local].trace_current[face].template data_ptr<double>();
            double* dst = buf;
            for (int64_t d : direction_blocks_[b].directions)
              for (int64_t g : group_indices_[s]) {
                const double* src = trace + (d * num_groups_ + g) * Nf;
                std::copy(src, src + Nf, dst);
                dst += Nf;
              }
            request = sweep_comm_.isend(buf, count, rank, tag);
            initiated = true;
            return task::TaskStatus::POLLING;
          }
          if (!request.test())
            return task::TaskStatus::POLLING;
          initiated = false;
          request = parallel::Request();
          return task::TaskStatus::COMPLETED;
        });
      send->add_dependency(tasks[task_of_unit_[a][ex.local]]);
    }
  }

  /// @brief Run the outer loop shared by both problem types.
  /// @param k_tol k convergence tolerance (eigenvalue mode only).
  void outer_iterations(double k_tol)
  {
    const int previous_threads = at::get_num_threads();
    if (config_.num_threads > 1)
      at::set_num_threads(1);

    const bool verbose = config_.verbose && comm_.rank() == 0;
    auto start = std::chrono::high_resolution_clock::now();
    double callback_seconds = 0.0;
    solve_seconds_ = 0.0;
    double previous_error = -1.0;
    if (eigenvalue_mode_) {
      // Normalize the initial flux to unit fission production with k = 1, so
      // the first iterate (fission source / k) has the scale of the warm
      // start; an initial k far from the eigenvalue makes a loosely solved
      // first inner (inner_forcing) meaningless.
      const double production = fission_production();
      for (auto& P : patches_)
        P.moments.div_(production);
      k_ = 1.0;
    }

    int64_t it = 0;
    for (; it < config_.max_iter; it++) {
      const auto old = scalar_flux_snapshot();
      if (eigenvalue_mode_) {
        // Fission source of the previous iterate, normalized by k
        thread_local std::vector<double> scratch;
        std::vector<int64_t> all(num_groups_);
        std::iota(all.begin(), all.end(), 0);
        for (auto& P : patches_) {
          if (!P.fission.defined())
            continue;
          P.fission_source.zero_();
          transfer_source(P.disc->view(), P.fission.template data_ptr<double>(),
            1, moment_order_raw_.data(), 1, num_groups_,
            P.moments.template data_ptr<double>(), 1.0 / k_, all.data(),
            num_groups_, nullptr, P.fission_source.template data_ptr<double>(),
            scratch);
        }
      }

      const double inner_tol = inner_tolerance();
      auto t0 = std::chrono::high_resolution_clock::now();
      outer_sweeps_.push_back(group_set_pass());
      sweep_times_.push_back(std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0)
          .count());

      double error, k_error = 0.0;
      if (eigenvalue_mode_) {
        // As TransportDriver::solve_eigenvalue(): relative to the old flux
        error = scalar_flux_change(old, true);
        const double k_new = fission_production();
        k_error = std::abs(k_new - k_);
        k_ = k_new;
        k_history_.push_back(k_);
      } else {
        error = scalar_flux_change(old, false);
      }
      errors_.push_back(error);
      const double rho = previous_error > 0 ? error / previous_error : 0.0;
      previous_error = error;
      double target = config_.tol;
      if (config_.rho_safety)
        target *= (1.0 - std::min(rho, 0.999));

      if (verbose) {
        std::cout << "-- outer (" << it << "): ";
        if (eigenvalue_mode_)
          std::cout << "k = " << std::fixed << std::setprecision(6) << k_
                    << ", dk = " << std::scientific << std::setprecision(2)
                    << k_error << ", ";
        std::cout << "scalar flux error = " << std::scientific
                  << std::setprecision(4) << error << ", rho = " << std::fixed
                  << std::setprecision(4) << rho
                  << ", group-set sweeps = " << outer_sweeps_.back()
                  << ", inner tol = " << std::scientific << std::setprecision(1)
                  << inner_tol << std::fixed
                  << ", elapsed = " << std::setprecision(3)
                  << std::chrono::duration<double>(
                       std::chrono::high_resolution_clock::now() - start)
                       .count()
                  << " s" << std::defaultfloat << std::endl;
      }
      solve_seconds_ = std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count() -
                       callback_seconds;
      if (callback_ && it % callback_frequency_ == 0) {
        auto c0 = std::chrono::high_resolution_clock::now();
        callback_(*this);
        callback_seconds += std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - c0)
                              .count();
      }
      const bool k_ok = !eigenvalue_mode_ || k_error < k_tol;
      if (it > 0 && error < target && k_ok) {
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
                << " (" << num_iterations_ << " outer iterations, "
                << equivalent_sweeps_ << " sweep equivalents)" << std::endl;
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
    const parallel::ParallelContext& mpi_context, bool serial = false)
    : mesh_(std::move(mesh)), xs_server_(std::move(xs_server)),
      comm_(serial ? parallel::Communicator::self()
                   : parallel::Communicator::world()),
      serial_(serial)
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
  /// @brief Use a fixed partition (GID -> MPI rank) for multi-rank sweeps
  /// instead of METIS. Takes effect at the next setup().
  void set_partition(std::unordered_map<int64_t, int> gid_to_rank)
  {
    partition_ = std::move(gid_to_rank);
  }

  /// @brief Build the patch discretizations, sweep graph and task DAG. On
  /// several MPI ranks the mesh must be complete on every rank: the global
  /// sweep graph is built first, then the mesh is partitioned (see
  /// set_partition()) and culled to this rank's patches.
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

    const bool distributed = comm_.size() > 1;
    if (distributed &&
        mesh_->get_blocks().size() != mesh_->get_global_num_blocks()) {
      throw utils::runtime_error(ctx("setup"),
        "Multi-rank sweeps need the full (undistributed) mesh on every rank: "
        "setup() builds the global sweep graph, then partitions the mesh");
    }

    std::tie(omega_, omega3_, weights_) = discrete_ordinates(*qset_, NumDim);
    const int64_t num_directions = omega_.size(0);

    // Every patch of the mesh is a node of the global sweep graph (cycles
    // must be broken consistently across ranks); only this rank's patches
    // are kept after partitioning
    std::vector<PatchState> nodes;
    std::unordered_map<int64_t, size_t> gid_to_node;
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
      if (mat.is_fissile()) {
        P.fission = torch::outer(mat.get_chi().to(torch::kFloat64).cpu(),
          mat.get_nu_fission().to(torch::kFloat64).cpu())
                      .unsqueeze(0)
                      .contiguous();
      }
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
      gid_to_node[P.gid] = nodes.size();
      nodes.push_back(std::move(P));
    }

    std::tie(harmonics_, moment_order_) =
      discrete_harmonics(*qset_, NumDim, num_orders);
    weighted_harmonics_ = (harmonics_ * weights_.unsqueeze(1)).contiguous();
    moment_order_raw_.assign(moment_order_.template data_ptr<int64_t>(),
      moment_order_.template data_ptr<int64_t>() + moment_order_.numel());
    harmonics_ = harmonics_.contiguous();
    omega_ = omega_.contiguous();

    // Face links and interfaces between nodes
    std::vector<SweepInterface> interfaces;
    std::map<std::pair<int64_t, int64_t>, int64_t> owner_face_to_interface;
    for (size_t i = 0; i < nodes.size(); i++) {
      auto& P = nodes[i];
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
            auto it = gid_to_node.find(conn.gid);
            if (it == gid_to_node.end()) {
              throw utils::runtime_error(
                ctx("setup"), "Neighbor patch " + std::to_string(conn.gid) +
                                " is not in the mesh");
            }
            link.neighbor = static_cast<int64_t>(it->second);
            link.neighbor_face = 2 * conn.dim + (conn.is_upper ? 1 : 0);
            link.neighbor_face_dofs =
              nodes[link.neighbor].disc->faces()[link.neighbor_face].num_dofs();
            link.trace_map =
              build_trace_map(dim, conn, *nodes[link.neighbor].disc)
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
    for (auto& P : nodes) {
      for (auto& link : P.links) {
        if (link.type == physics::BoundaryType::INTERNAL && !link.is_first) {
          link.interface =
            owner_face_to_interface.at({link.neighbor, link.neighbor_face});
        }
      }
    }
    graph_ = SweepGraph::create(static_cast<int64_t>(nodes.size()),
      std::move(interfaces), num_directions);

    // Partition, keep this rank's patches and turn links to other ranks'
    // patches into ghost links
    node_rank_ = partition(nodes);
    node_to_local_.assign(nodes.size(), -1);
    local_to_node_.clear();
    for (size_t node = 0; node < nodes.size(); node++) {
      if (node_rank_[node] != comm_.rank())
        continue;
      node_to_local_[node] = static_cast<int64_t>(patches_.size());
      local_to_node_.push_back(static_cast<int64_t>(node));
      gid_to_local_[nodes[node].gid] = patches_.size();
      patches_.push_back(std::move(nodes[node]));
    }
    nodes.clear();
    for (auto& P : patches_) {
      for (auto& link : P.links) {
        if (link.type != physics::BoundaryType::INTERNAL)
          continue;
        const int64_t nb = link.neighbor;
        if (node_to_local_[nb] >= 0) {
          link.neighbor = node_to_local_[nb];
        } else {
          link.neighbor = -1;
          link.remote_rank = node_rank_[nb];
        }
      }
    }
    if (distributed)
      sweep_comm_ = comm_.duplicate();

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
    group_masks_.clear();
    group_complements_.clear();
    for (const auto& g : group_sets_) {
      group_indices_.emplace_back(g.template data_ptr<int64_t>(),
        g.template data_ptr<int64_t>() + g.numel());
      std::vector<uint8_t> mask(num_groups_, 0), complement(num_groups_, 1);
      for (int64_t k : group_indices_.back()) {
        mask[k] = 1;
        complement[k] = 0;
      }
      group_masks_.push_back(std::move(mask));
      group_complements_.push_back(std::move(complement));
    }

    // Lagged traces: reflective faces (every direction is read for its
    // reflection) and interface faces read in a LAGGED state
    // Key: (unit, face, ghost) of the buffer holding the lagged trace
    std::map<std::tuple<int64_t, int64_t, bool>, std::set<int64_t>> lagged;
    for (size_t i = 0; i < patches_.size(); i++) {
      const auto& P = patches_[i];
      for (size_t f = 0; f < P.links.size(); f++) {
        const auto& link = P.links[f];
        if (link.type == physics::BoundaryType::REFLECTIVE) {
          auto& dirs = lagged[{static_cast<int64_t>(i), f, false}];
          for (int64_t d = 0; d < num_directions; d++)
            dirs.insert(d);
        } else if (link.type == physics::BoundaryType::INTERNAL) {
          const auto& sets = graph_->angle_sets();
          for (size_t a = 0; a < sets.size(); a++) {
            const EdgeState state = link.is_first
                                      ? sets[a].backward[link.interface]
                                      : sets[a].forward[link.interface];
            if (state != EdgeState::LAGGED)
              continue;
            auto& dirs = link.remote_rank >= 0
                           ? lagged[{static_cast<int64_t>(i), f, true}]
                           : lagged[{link.neighbor, link.neighbor_face, false}];
            for (int64_t d : set_directions_[a])
              dirs.insert(d);
          }
        }
      }
    }
    delayed_.clear();
    for (const auto& [key, dirs] : lagged) {
      delayed_.push_back({std::get<0>(key), std::get<1>(key),
        std::vector<int64_t>(dirs.begin(), dirs.end()), std::get<2>(key)});
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
      P.ghost_current.assign(P.links.size(), torch::Tensor());
      P.ghost_previous.assign(P.links.size(), torch::Tensor());
      for (size_t f = 0; f < P.links.size(); f++) {
        if (P.links[f].remote_rank >= 0) {
          P.ghost_current[f] = torch::zeros(
            {num_directions, num_groups_, P.links[f].neighbor_face_dofs},
            torch::kFloat64);
          P.ghost_previous[f] = torch::zeros_like(P.ghost_current[f]);
        }
      }
      P.moments = torch::zeros({M, num_groups_, N}, torch::kFloat64);
      P.moments_next = torch::zeros_like(P.moments);
      P.source_moments = torch::zeros_like(P.moments);
      P.krylov_moments = torch::zeros_like(P.moments);
      P.pass_moments = torch::zeros_like(P.moments);
      P.fission_source = P.fission.defined()
                           ? torch::zeros({num_groups_, N}, torch::kFloat64)
                           : torch::Tensor();
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
      P.factors_ready.assign(group_sets_.size(), 0);
      P.psi =
        config_.store_angular_flux
          ? torch::zeros({num_directions, num_groups_, N}, torch::kFloat64)
          : torch::Tensor();
    }

    scheduler_ = std::make_unique<task::TaskScheduler>(
      std::max<int64_t>(1, config_.num_threads));
    build_dag();
    build_range_dag();
    is_setup_ = true;
  }

  /// @brief One classic source-iteration step over every group set: every
  /// set sees the moments from before the step (Jacobi) and is swept once.
  void sweep()
  {
    if (!is_setup_)
      throw utils::runtime_error(ctx("sweep"), "Call setup() first");
    for_each_patch([](PatchState& P) { P.pass_moments.copy_(P.moments); });
    for (size_t s = 0; s < group_sets_.size(); s++) {
      compute_source(s, Within::MOMENTS, true, true);
      sweep_group_set(s);
      commit_group_set(s);
    }
  }

  /// @brief Solve a fixed-source problem.
  /// @return The solution: the angular flux if `store_angular_flux`, else
  /// the scalar flux.
  typename Solution::Ptr solve_fixed_source()
  {
    if (!is_setup_)
      throw utils::runtime_error(
        ctx("solve_fixed_source"), "Call setup() first");
    if (has_fission_)
      throw utils::runtime_error(ctx("solve_fixed_source"),
        "Fissile materials need solve_eigenvalue(); subcritical "
        "multiplication (fixed source + fission) is not supported");
    eigenvalue_mode_ = false;
    reset_state(false);
    outer_iterations(0.0);
    return build_solution();
  }

  /// @brief Solve the k-eigenvalue problem by power iteration, mirroring
  /// TransportDriver::solve_eigenvalue(): start from a unit scalar flux, put
  /// F phi / k on the right-hand side of each across-group-set pass, and
  /// update k from the fission production.
  /// @param k_tol Tolerance on |k_n - k_{n-1}| (with config.tol on the
  /// scalar flux change relative to the previous iterate).
  /// @return The solution, holding k_eff.
  typename Solution::Ptr solve_eigenvalue(double k_tol = 1e-6)
  {
    if (!is_setup_)
      throw utils::runtime_error(ctx("solve_eigenvalue"), "Call setup() first");
    if (!has_fission_)
      throw utils::runtime_error(
        ctx("solve_eigenvalue"), "No fissile material on this rank");
    for (const auto& P : patches_)
      if (P.external.defined())
        throw utils::runtime_error(ctx("solve_eigenvalue"),
          "Fixed sources are not supported in eigenvalue problems");
    eigenvalue_mode_ = true;
    reset_state(true);
    outer_iterations(k_tol);
    return build_solution();
  }

  /// @brief Wrap the current flux in a TransportSolution so the existing
  /// error and comparison tooling applies.
  typename Solution::Ptr build_solution() const
  {
    const bool angular = config_.store_angular_flux;
    auto solution = Solution::create(serial_ ? parallel::Communicator::self()
                                             : parallel::Communicator::world(),
      mesh_, qset_, xs_server_, physics::DGTransportAssemblerConfig(),
      eigenvalue_mode_ ? std::optional<double>(k_) : std::nullopt, angular);
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

  /// @brief Set a callback invoked after every `frequency`-th outer
  /// iteration of solve_fixed_source()/solve_eigenvalue() (as
  /// TransportDriver::set_callback()). Time spent in the callback is
  /// excluded from get_solve_seconds(). Pass an empty function to disable.
  void set_callback(Callback callback, int64_t frequency = 1)
  {
    if (frequency < 1) {
      throw utils::runtime_error(
        ctx("set_callback"), "`frequency` must be at least 1");
    }
    callback_ = std::move(callback);
    callback_frequency_ = frequency;
  }

  // =================================================================
  // Public getters
  /// @return Wall seconds of the running/last solve up to its last
  /// completed outer iteration, excluding callback time.
  double get_solve_seconds() const noexcept { return solve_seconds_; }
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
  /// @return k of the last eigenvalue solve, and its history.
  double get_k_eff() const noexcept { return k_; }
  const std::vector<double>& get_k_history() const noexcept
  {
    return k_history_;
  }
  /// @return Group-set sweeps of each outer iteration of the last solve.
  const std::vector<int64_t>& get_outer_sweeps() const noexcept
  {
    return outer_sweeps_;
  }
  /// @return Group-set sweeps of the last solve.
  int64_t get_group_set_sweeps() const noexcept { return group_set_sweeps_; }
  /// @return Full-sweep equivalents of the last solve (sum of |set| / G).
  double get_equivalent_sweeps() const noexcept { return equivalent_sweeps_; }
  /// @return Number of lagged face-trace blocks in the GMRES unknown.
  int64_t num_delayed_blocks() const noexcept
  {
    return static_cast<int64_t>(delayed_.size());
  }
  bool is_converged() const noexcept { return converged_; }
  /// @return Cumulative seconds spent inside sweep task bodies, summed over
  /// worker threads (compare with sweep wall time to see scheduler cost).
  double get_task_seconds() const noexcept
  {
    return static_cast<double>(task_nanoseconds_.load()) * 1e-9;
  }
  /// @return Cumulative wall seconds spent executing sweep DAGs; the rest of
  /// a sweep's wall time is serial work on the calling thread.
  double get_dag_seconds() const noexcept
  {
    return static_cast<double>(dag_nanoseconds_) * 1e-9;
  }
  void reset_task_seconds() noexcept
  {
    task_nanoseconds_ = 0;
    dag_nanoseconds_ = 0;
  }
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
  /// @return Tasks of one sweep over every group set.
  int64_t num_tasks() const noexcept
  {
    int64_t n = 0;
    for (const auto& dag : dags_)
      n += static_cast<int64_t>(dag->size());
    return n;
  }
  const typename Mesh::Ptr& get_mesh() const noexcept { return mesh_; }
};

} // namespace ttnte::sweep
