#pragma once

#include "ttnte/cad/interface_quadrature.hpp"
#include "ttnte/cad/patch.hpp"
#include "ttnte/cross/layout_cross.hpp"
#include "ttnte/driver/transport_solution.hpp"
#include "ttnte/linalg/banded_cholesky.hpp"
#include "ttnte/linalg/krylov.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/heuristics.hpp"
#include "ttnte/parallel/load_balancer.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/physics/boundary_types.hpp"
#include "ttnte/sweep/angular_moments.hpp"
#include "ttnte/sweep/fiber_block.hpp"
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
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ttnte::sweep {

/// @brief Discrete-ordinates transport sweep solver for discontinuous
/// multi-patch IGA (dimension a runtime value; see SweepDriver for the typed
/// front end the bindings use).
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
class SweepSolver {
public:
  // =================================================================
  // Public types
  using Mesh = mesh::Mesh<cad::Patch>;
  using Ptr = std::shared_ptr<SweepSolver>;
  using Solution = driver::TransportSolution<cad::Patch>;
  /// Called after every `frequency`-th outer iteration of a solve.
  using Callback = std::function<void(const SweepSolver&)>;

  using FiberBlock = sweep::FiberBlock;

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
    /// INTERNAL: (Nf) neighbor-face index of each of this face's dofs
    /// (conforming interfaces).
    torch::Tensor trace_map;
    /// INTERNAL, non-conforming: (quadrature point, own face dof, neighbor
    /// face dof, R_a R^nb_b) pairs at the shared interface quadrature.
    torch::Tensor coupling_q;
    torch::Tensor coupling_a;
    torch::Tensor coupling_b;
    torch::Tensor coupling_val;
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
    typename cad::Patch::Ptr block;
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
    /// sweep (undefined if the face has no buffer; the previous one only
    /// exists where a delayed block reads it).
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
    /// Kronecker-sum solver (SYLVESTER on an affine patch), or null.
    KroneckerSolver::Ptr kronecker;
    /// Partially separable solver (SYLVESTER on an extruded patch), or null.
    ExtrudedSolver::Ptr extruded;
    /// Matrix-free iterative solver (ITERATIVE on a curved patch), or null.
    IterativeSolver::Ptr iterative;
    /// Factorization cache (see SweepConfig::cache_factorizations).
    torch::Tensor factors;
    torch::Tensor factor_pivots;
    /// Per (direction, group): whether the cache holds its factors (filled
    /// by whichever sweep, exact or fiber, factors the pair first).
    std::vector<uint8_t> factor_ready;
    /// Guards moments_next accumulation across concurrent angle sets.
    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();
  };

  // =================================================================
  // Private data
  /// Spatial dimension of the mesh (1 to 3).
  int64_t ndim_ = 0;
  typename Mesh::Ptr mesh_;
  xs::Server::Ptr xs_server_;
  parallel::Communicator comm_;
  math::QuadratureSet::Ptr qset_;
  SweepConfig config_;

  /// (N_Omega, ndim_) streaming directions, (N_Omega, 3) unit vectors and
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

  /// One task family of a fiber sweep: directions of one angle set and a
  /// chunk of groups, with the output slot of each (direction, first group).
  struct FiberTask {
    size_t set = 0;
    std::vector<int64_t> directions;
    std::vector<int64_t> groups;
    std::vector<int64_t> slots;
  };
  std::vector<FiberTask> fiber_tasks_;
  /// Offset of each local unit in a fiber (units concatenated in local
  /// order), and the fiber length.
  std::vector<int64_t> fiber_offsets_;
  int64_t fiber_length_ = 0;

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
  /// each node (-1 if remote).
  std::vector<int> node_rank_;
  std::vector<int64_t> node_to_local_;

  /// One sweep DAG per group set.
  std::vector<std::unique_ptr<task::TaskGraph>> dags_;
  std::unique_ptr<task::TaskScheduler> scheduler_;
  /// Flat DAG of index-range tasks running range_fn_ (see parallel_for()).
  std::unique_ptr<task::TaskGraph> range_dag_;
  const std::function<void(int64_t)>* range_fn_ = nullptr;
  int64_t range_size_ = 0;
  bool is_setup_ = false;
  /// Whether any material of the (global) mesh is fissile.
  bool has_fission_ = false;
  /// Whether any patch of the (global) mesh has a fixed source.
  bool has_external_ = false;
  /// Cumulative time spent inside sweep task bodies (all threads).
  std::atomic<int64_t> task_nanoseconds_ {0};
  /// Wall time spent executing sweep DAGs (the parallel part of a sweep).
  int64_t dag_nanoseconds_ = 0;
  /// Fibers swept by sweep_fibers().
  int64_t num_fibers_ = 0;
  /// TransportSolve::CROSS: one cross per group set, the tolerance of the
  /// next transport solve, and per-solve history (tolerance, fibers, largest
  /// rank, hold-out error).
  std::vector<cross::LayoutCross::Ptr> crosses_;
  double cross_tol_ = 0.0;
  /// SweepConfig::correction_form: ||b|| / ||x|| of each group set from its
  /// last standard within-group-set solve (b the transport solve of the fixed
  /// sources, x the solution), which turns tolerances relative to ||b|| into
  /// tolerances relative to ||y0|| ~ ||x|| (0: not measured yet).
  std::vector<double> rhs_ratio_;
  /// SweepConfig::incremental_transport: per group set, the last full input
  /// (Krylov vector and source moments of the set) and its transport y.
  std::vector<std::vector<double>> anchor_x_;
  std::vector<std::vector<double>> anchor_y_;
  std::vector<std::vector<double>> anchor_source_;
  /// Increments since the last full transport solve of each group set, and
  /// the totals of incremental and full transport solves of y0.
  std::vector<int64_t> anchor_steps_;
  int64_t incremental_solves_ = 0;
  int64_t full_solves_ = 0;
  std::vector<double> cross_tolerances_;
  std::vector<int64_t> cross_fibers_;
  std::vector<int64_t> cross_max_ranks_;
  std::vector<double> cross_errors_;
  /// Per transport solve: the ranks of the TT and the per-sweep history.
  std::vector<std::vector<int64_t>> cross_ranks_;
  std::vector<std::vector<cross::LayoutCross::SweepRecord>> cross_histories_;
  /// Wall time of the crosses and of their fiber sweeps (last solve).
  double cross_seconds_ = 0.0;
  double cross_sweep_seconds_ = 0.0;
  /// Seconds per phase of the cross transport solves outside LayoutCross::run
  /// (moments, traces, angular flux).
  std::map<std::string, double> cross_timings_;
  /// Seconds per step of the within-group-set solves (see solve_timings()).
  std::map<std::string, double> solve_timings_;

  /// Optional per-outer-iteration callback and its frequency.
  Callback callback_;
  int64_t callback_frequency_ = 1;
  /// Wall time of the running solve at the last completed outer iteration,
  /// excluding time spent in the callback.
  double solve_seconds_ = 0.0;
  /// Start of the running solve and time of the last progress line.
  std::chrono::steady_clock::time_point solve_start_;
  std::chrono::steady_clock::time_point last_progress_;

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
  /// SweepConfig::certify: scalar-flux change (relative L2) and change in k
  /// of the exact transport solve after the outer iteration (NaN if none),
  /// and its wall time.
  double certification_error_ = std::numeric_limits<double>::quiet_NaN();
  double certification_k_change_ = std::numeric_limits<double>::quiet_NaN();
  double certification_seconds_ = 0.0;
  /// Group-set sweeps of the last solve, and the same in full-sweep
  /// equivalents (a group-set sweep counts |set| / G).
  int64_t group_set_sweeps_ = 0;
  double equivalent_sweeps_ = 0.0;

  // =================================================================
  // Private helpers
  [[nodiscard]] static std::string ctx(const std::string& f)
  {
    return "ttnte::sweep::SweepSolver::" + f;
  }

  /// @return (Nf) index of the neighbor face dof for each dof of this face,
  /// mirroring NeighborCoupling::set_recv_buffer() (permute, then flip the
  /// perpendicular dims).
  torch::Tensor build_trace_map(size_t dim, const mesh::NeighborInfo& conn,
    const PatchDiscretization& nb) const;

  /// @return (N_Omega) index of the specular reflection of each direction on
  /// an axis-aligned face.
  torch::Tensor build_reflection(const FaceGeometry& face) const;

  /// @return The trace buffer of a lagged block (its current or previous
  /// sweep).
  const torch::Tensor& delayed_buffer(
    const DelayedBlock& blk, bool current) const;
  torch::Tensor& delayed_buffer(const DelayedBlock& blk, bool current)
  {
    // The object is not const here, so neither is the buffer
    return const_cast<torch::Tensor&>(
      std::as_const(*this).delayed_buffer(blk, current));
  }

  /// @return The sum of `value` over all ranks.
  double global_sum(double value) const;

  /// @brief Inflow of face `f` of unit `local` for an angle set.
  FaceInflow face_inflow(const AngleSet& set, size_t local, size_t f) const;

  /// @brief One sweep task: sweep the units of chunk `c` of angle set
  /// `set_index` for group set `group_set`, in upwind order.
  /// @return The sweep block of `directions` x `groups` (global indices).
  SweepBlock make_block(const std::vector<int64_t>& directions,
    const std::vector<int64_t>& groups) const;

  /// @return The current flux, angular or scalar, as a TransportSolution.
  typename Solution::Ptr make_solution(bool angular) const;

  /// @return (N_Omega, |group set s|, N) angular flux of unit `local` from
  /// group set s's cross.
  torch::Tensor cross_patch_psi(size_t s, size_t local) const;

  /// @brief Sets the ATen intra-op thread count for a scope (0 leaves it)
  /// and restores it on exit, also when a task throws.
  struct ThreadCountGuard {
    explicit ThreadCountGuard(int64_t num_threads)
      : previous(at::get_num_threads())
    {
      if (num_threads > 0)
        at::set_num_threads(static_cast<int>(num_threads));
    }
    ~ThreadCountGuard() { at::set_num_threads(previous); }
    ThreadCountGuard(const ThreadCountGuard&) = delete;
    ThreadCountGuard& operator=(const ThreadCountGuard&) = delete;
    int previous;
  };

  /// @return Unit `local` swept for angle set `set`: its operators, source,
  /// inflow and outgoing traces (outputs and factor cache left to the
  /// caller).
  UnitSweep make_unit(const AngleSet& set, int64_t local);

  /// @brief Point `unit` at patch P's factorization cache, if it has one.
  static void use_factor_cache(PatchState& P, UnitSweep& unit);

  /// @return This worker thread's sweep scratch space.
  static SweepWorkspace& workspace()
  {
    thread_local SweepWorkspace work;
    return work;
  }

  /// @brief One sweep task: sweep the units of chunk `c` (in upwind order)
  /// for the directions of a direction block at the groups of a group set,
  /// storing psi when the patch keeps it and adding the units' moment
  /// contributions into `moments_next` under the patch's mutex (blocks of
  /// different angle sets run concurrently).
  /// @param block_index Index into `direction_blocks_`.
  /// @param group_set Index into `group_indices_`.
  /// @param c Chunk of the block's angle-set schedule.
  void run_chunk(size_t block_index, size_t group_set, size_t c);

  /// @brief One fiber-sweep task: sweep the units of chunk `c` of task
  /// family `t`, writing their solutions into `out`.
  void run_fiber_chunk(size_t t, size_t c, double* out);

  /// @return Directions swept over all groups to start a cross: per
  /// quadrant/octant of the streaming components, the one closest to the
  /// octant diagonal of the unit sphere, judged on all three direction
  /// cosines (in 2-D the out-of-plane one too, so a direction nearly normal
  /// to the plane is not taken). 1-D: the most and least grazing of each
  /// sign.
  std::vector<int64_t> initial_directions() const;

  /// @brief One cross per group set (TransportSolve::CROSS).
  void build_crosses();

  /// @brief Transport solve of group set `s` by the cross: moments into
  /// moments_next, and the lagged and reflective traces (the delayed
  /// blocks) and the angular flux, if stored, from the low-rank flux.
  void cross_group_set(size_t s);

  /// @brief Run `fn` and add its wall time to solve_timings_[phase].
  template<typename Fn>
  void timed(const char* phase, Fn&& fn)
  {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    solve_timings_[phase] +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
  }

  /// @brief One transport solve of group set `s` with the current sources
  /// and lagged traces: an exact sweep or a cross.
  void transport_group_set(size_t s);

  /// Which moments feed the within-group-set scattering term.
  enum class Within : uint8_t { NONE, MOMENTS, KRYLOV };

  /// @brief Call `fn(i)` for i in [0, n), in parallel on the scheduler's
  /// workers (`fn` must be safe for distinct i; it may use thread_local
  /// scratch). Serial with one thread.
  void parallel_for(int64_t n, const std::function<void(int64_t)>& fn);

  /// @brief Apply `fn` to every patch in parallel (see parallel_for()).
  void for_each_patch(const std::function<void(PatchState&)>& fn)
  {
    parallel_for(static_cast<int64_t>(patches_.size()),
      [&](int64_t i) { fn(patches_[i]); });
  }

  /// @brief Build the range tasks of parallel_for(): task c of K handles
  /// [c n / K, (c + 1) n / K). None (serial) with one thread.
  void build_range_dag();

  /// @brief Source moments of group set `s` for every unit.
  /// @param within Moments of the within-set scattering term (NONE leaves it
  /// out, KRYLOV uses the GMRES iterate).
  /// @param fixed Include the parts that do not depend on this set's flux:
  /// external and fission sources, and scattering from the other sets.
  /// @param jacobi Take the other sets' flux from the start of the pass.
  void compute_source(size_t s, Within within, bool fixed, bool jacobi);

  /// @brief Sweep group set `s` with the current source moments and lagged
  /// traces, accumulating its moments into moments_next.
  void sweep_group_set(size_t s);

  /// @brief Accept the last sweep of group set `s`: moments and outgoing
  /// traces become the current iterate.
  void commit_group_set(size_t s);

  /// @return Length of the GMRES unknown of group set `s`.
  int64_t krylov_size(size_t s) const;

  /// @brief Gather the moments of group set `s` (from moments or
  /// moments_next) and its lagged traces (from the previous or current
  /// traces) into a Krylov vector.
  void pack(size_t s, bool from_next, bool from_current, double* x) const;

  /// @brief Scatter a Krylov vector of group set `s` into the moments (the
  /// GMRES iterate or the current moments) and the lagged (previous)
  /// traces.
  void unpack(size_t s, const double* x, bool to_krylov);

  /// @brief Print a progress line of the within-group-set iteration when
  /// SweepConfig::progress_seconds have passed since the last one.
  void report_progress(size_t s, int64_t iteration, double residual,
    const char* method, bool force = false);

  /// @return Within-group-set tolerance of the current outer iteration.
  double inner_tolerance() const;

  /// @return Global inner product of two rank-local slices.
  static double global_dot(const std::vector<double>& a,
    const std::vector<double>& b, const linalg::krylov::Reducer& reducer);

  /// @brief GMRES on (I - T S) x = rhs within group set `s`, every product a
  /// transport solve whose CROSS accuracy is relaxed as the residual drops.
  /// @param scale Ratio of ||rhs|| to the norm residuals are reported and
  /// relaxed against (1 for the standard form, ||r0|| / ||y0|| for the
  /// correction form); `tol` is relative to that norm.
  /// @param x Initial guess on entry, solution on exit.
  void within_gmres(size_t s, bool jacobi, const std::vector<double>& rhs,
    std::vector<double>& x, double tol, double tol_min, double scale,
    const linalg::krylov::Reducer& reducer);

  /// @brief Within-group-set GMRES for the correction to the current
  /// iterate (SweepConfig::correction_form).
  /// @param s Group set.
  /// @param jacobi Take the other sets' flux from the start of the pass.
  /// @param tol Within-group-set tolerance, relative to ||y0||.
  /// @param tol_min Accuracy of a CROSS transport solve of the full source.
  /// @param x Current iterate on entry (packed moments and lagged traces);
  /// the accepted iterate is written to the moments and lagged traces.
  /// @param reducer Global sum of inner products (empty in serial).
  void correction_solve(size_t s, bool jacobi, double tol, double tol_min,
    std::vector<double>& x, const linalg::krylov::Reducer& reducer);

  /// @brief Store the source moments of group set `s` in `saved` and, when
  /// `difference`, replace them by their change from the previous `saved`.
  void swap_source(size_t s, std::vector<double>& saved, bool difference);

  /// @brief Rank-local squared norms of the change of group set `s`'s source
  /// moments since `saved` (see swap_source()) and of the source itself.
  void source_change(size_t s, const std::vector<double>& saved,
    double& change_sq, double& source_sq) const;

  /// @brief y0 = T(input) as y_prev + T(input - input_prev) when the change
  /// since the last full transport solve of group set `s` is small
  /// (SweepConfig::incremental_transport). The source moments of the
  /// current input must be computed.
  /// @return False (nothing done) when a full transport solve is needed.
  bool try_incremental_transport(size_t s, const std::vector<double>& x,
    double tol_min, std::vector<double>& y,
    const linalg::krylov::Reducer& reducer);

  /// @return SweepConfig::global_inner_tolerance: ||phi|| / ||phi_s|| of the
  /// current scalar flux over all groups and over group set `s`, in the
  /// norm of the outer test (the unweighted coefficient norm, not the L2
  /// norm of the flux), 1 while the set has no flux. With fission, the smaller
  /// of that and P / P_s, P_s the set's share of the fission production, so a
  /// set that makes neutrons stays tight however small its flux; a set
  /// without fission keeps the flux ratio (its error reaches the other sets
  /// through scattering).
  double global_tolerance_scale(size_t s) const;

  /// @brief Solve group set `s` for its within-set scattering.
  /// @return Group-set sweeps used.
  int64_t inner_solve(size_t s, bool jacobi);

  /// @brief One across-group-set pass (Gauss-Seidel or Jacobi).
  /// @return Group-set sweeps used.
  int64_t group_set_pass();

  /// @return The scalar flux (moment 0) of every unit, concatenated.
  std::vector<double> scalar_flux_snapshot() const;

  /// @return ||phi - old|| / ||denominator|| over the scalar flux.
  double scalar_flux_change(
    const std::vector<double>& old, bool relative_to_old) const;

  /// @return Total fission production of the current flux.
  double fission_production() const;

  /// @brief Reset the iterate: zero traces, and zero or unit scalar flux.
  void reset_state(bool unit_scalar_flux);

  /// @brief Assign every node to a rank (the user partition, else METIS on
  /// the patch connectivity graph weighted by patch dofs) and cull the mesh
  /// to this rank's blocks. The caller's mesh stays culled: a later setup()
  /// on more than one rank needs a fresh full mesh.
  /// @return The rank of every node.
  /// TODO: (1) a measured per-patch sweep time would weigh patches better
  /// than their dofs (local solver cost differs, e.g. curved patches).
  /// (2) Edge weights (communication): the number of face dofs per
  /// interface. Without them METIS treats a tiny face and a large face as
  /// equal cuts.
  std::vector<int> partition(const std::vector<PatchState>& nodes);

  /// @brief Build one sweep DAG per group set: one task per (direction
  /// block, chunk), plus MPI receive/send tasks for cross-rank interfaces.
  void build_dag();

  /// @brief One task per chunk of angle set `set`'s schedule, depending on
  /// its upstream chunks, each running `body(chunk)`.
  /// @return The tasks, by chunk.
  std::vector<task::Task*> add_chunk_tasks(task::TaskGraph& dag, size_t set,
    const std::string& label, std::function<void(size_t)> body);

  /// @brief MPI tasks of direction block `b` for group set `s`: receive the
  /// remote upstream traces into ghosts (the local sweep task waits for
  /// DEPENDENCY edges) and send the local outgoing traces after their task.
  /// Messages carry the block's directions x the set's groups x face dofs.
  void add_exchange_tasks(task::TaskGraph& dag,
    const std::vector<task::Task*>& tasks, size_t b, size_t s);

  /// @brief Fission source of the current iterate, normalized by k.
  void update_fission_source();

  /// @brief SweepConfig::certify: one exact transport solve of every group
  /// set at the converged sources. Jacobi over the sets, so every set sees
  /// the same converged flux; the exact moments and traces then replace the
  /// cross's, and the scalar-flux change (and the change in k) is the
  /// certification error.
  void certify();

  /// @brief Run the outer loop shared by both problem types.
  /// @param k_tol k convergence tolerance (eigenvalue mode only).
  void outer_iterations(double k_tol);

public:
  // =================================================================
  // Public constructors
  virtual ~SweepSolver() = default;

protected:
  /// @brief Construct a SweepSolver object.
  /// @param num_dim Spatial dimension (1 to 3) of the mesh.
  /// @param mesh The (finalized or finalizable) mesh; every block local to
  /// this rank is a sweep unit.
  /// @param xs_server Multigroup cross sections.
  /// @param mpi_context The MPI context (same argument as TransportDriver).
  /// @param serial Which MPI communicator to use.
  SweepSolver(int64_t num_dim, Mesh::Ptr mesh, xs::Server::Ptr xs_server,
    const parallel::ParallelContext& mpi_context, bool serial = false);

public:
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
    const std::unordered_map<int64_t, int64_t>& chunks = {});

  /// @brief Sweep the fibers of `blocks` with the current source moments
  /// and lagged traces, without changing the iterate: each (direction,
  /// group) pair is one exact upwind sweep through every local unit.
  ///
  /// The fiber of block k, direction i and group j is row
  /// offset_k + i * |groups_k| + j of `out`, offset_k being the number of
  /// fibers of the blocks before k. A row holds the local units' dofs in
  /// local order (see fiber_offsets()). Pairs must not repeat (checked). The
  /// interface traces of the swept pairs are overwritten.
  /// @param blocks The fibers to sweep.
  /// @param out (num_fibers, fiber_length()) output.
  /// @param done Optional: called once for every range of finished rows
  /// (one task per direction run and group chunk, after its last unit), from
  /// the scheduler's workers, so work on finished fibers overlaps the rest
  /// of the sweep.
  void sweep_fibers(const std::vector<FiberBlock>& blocks, double* out,
    const FiberDone& done = {});

  /// @return Length of a fiber (dofs of every local unit).
  int64_t fiber_length() const noexcept { return fiber_length_; }
  /// @return Offset of each local unit's dofs in a fiber.
  const std::vector<int64_t>& fiber_offsets() const noexcept
  {
    return fiber_offsets_;
  }
  /// @return The cross of group set `s` (TransportSolve::CROSS), holding
  /// the TT of the last transport solve.
  cross::LayoutCross::Ptr get_cross(size_t s = 0) const;
  /// @return Cores of patch `gid`'s angular flux TT for group set `s` (last
  /// transport solve), in the configured layout.
  std::vector<torch::Tensor> get_patch_angular_flux_tt(
    int64_t gid, size_t s = 0) const
  {
    return get_cross(s)->patch_tt(gid_to_local_.at(gid));
  }
  const std::vector<double>& get_cross_tolerances() const noexcept
  {
    return cross_tolerances_;
  }
  const std::vector<int64_t>& get_cross_fibers() const noexcept
  {
    return cross_fibers_;
  }
  const std::vector<int64_t>& get_cross_max_ranks() const noexcept
  {
    return cross_max_ranks_;
  }
  double get_cross_seconds() const noexcept { return cross_seconds_; }
  /// @return Seconds per phase of the cross transport solves: the phases of
  /// LayoutCross::run summed over the group sets' crosses (see
  /// LayoutCross::timings()), and moments, traces and angular_flux.
  std::map<std::string, double> get_cross_timings() const;
  /// @return Seconds per step of the within-group-set solves, accumulated:
  /// inner (whole within-group-set solves), gmres (the Krylov solves within
  /// them), and unpack, source, transport and pack per transport solve.
  const std::map<std::string, double>& get_solve_timings() const noexcept
  {
    return solve_timings_;
  }
  double get_cross_sweep_seconds() const noexcept
  {
    return cross_sweep_seconds_;
  }
  const std::vector<std::vector<int64_t>>& get_cross_ranks() const noexcept
  {
    return cross_ranks_;
  }
  const std::vector<std::vector<cross::LayoutCross::SweepRecord>>&
  get_cross_histories() const noexcept
  {
    return cross_histories_;
  }
  const std::vector<double>& get_cross_errors() const noexcept
  {
    return cross_errors_;
  }
  /// @return GID of every local unit, in local (fiber) order.
  std::vector<int64_t> local_gids() const;
  /// @return Fibers swept by sweep_fibers() since setup.
  int64_t num_fibers() const noexcept { return num_fibers_; }

  /// @brief One classic source-iteration step over every group set: every
  /// set sees the moments from before the step (Jacobi) and is swept once.
  void sweep();

  /// @brief Solve a fixed-source problem.
  /// @return The solution: the angular flux if `store_angular_flux`, else
  /// the scalar flux.
  typename Solution::Ptr solve_fixed_source();

  /// @brief Solve the k-eigenvalue problem by power iteration, mirroring
  /// TransportDriver::solve_eigenvalue(): start from a unit scalar flux, put
  /// F phi / k on the right-hand side of each across-group-set pass, and
  /// update k from the fission production.
  /// @param k_tol Tolerance on |k_n - k_{n-1}| (with config.tol on the
  /// scalar flux change relative to the previous iterate).
  /// @return The solution, holding k_eff.
  typename Solution::Ptr solve_eigenvalue(double k_tol = 1e-6);

  /// @brief Wrap the current flux in a TransportSolution so the existing
  /// error and comparison tooling applies: angular if it is stored
  /// (config.store_angular_flux), scalar otherwise.
  typename Solution::Ptr build_solution() const
  {
    return make_solution(config_.store_angular_flux);
  }

  /// @brief The solution with its angular dependence, from the stored
  /// angular flux or, with transport = CROSS and cross.keep_angular_flux, from
  /// the TT of the last transport solve (e.g. for particle balances).
  typename Solution::Ptr angular_solution() const
  {
    return make_solution(true);
  }

  /// @return The scalar flux of a patch, shape (*ctrlpts_grid, G).
  torch::Tensor get_scalar_flux(int64_t gid) const;

  /// @return The flux moments of a patch, shape (num_moments, G, N).
  const torch::Tensor& get_moments(int64_t gid) const
  {
    return patches_.at(gid_to_local_.at(gid)).moments;
  }

  /// @return The angular flux of a patch in the TT core order,
  /// (*angular_modes, *ctrlpts_grid, G).
  torch::Tensor get_angular_flux(int64_t gid) const;

  /// @return The factored angular mode sizes (TT core order).
  std::vector<int64_t> angular_modes() const;

  /// @brief Dense matrix of a patch's full local operator with reflective
  /// faces implicit and internal faces excluded -- the operator the TT
  /// assembler stores as the LinearSystem's interior operator -- in the TT
  /// ordering (direction, dofs, group) for rows and columns. For
  /// verification on small problems only.
  /// @return (N_Omega * N * G, N_Omega * N * G) dense matrix.
  torch::Tensor assemble_dense_operator(int64_t gid) const;

  /// @brief Set a callback invoked after every `frequency`-th outer
  /// iteration of solve_fixed_source()/solve_eigenvalue() (as
  /// TransportDriver::set_callback()). Time spent in the callback is
  /// excluded from get_solve_seconds(). Pass an empty function to disable.
  void set_callback(Callback callback, int64_t frequency = 1);

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
  /// @return SweepConfig::incremental_transport: y0 solves by increment and
  /// by a full transport solve since the last solve started.
  int64_t get_incremental_solves() const noexcept
  {
    return incremental_solves_;
  }
  int64_t get_full_solves() const noexcept { return full_solves_; }
  /// @return Full-sweep equivalents of the last solve (sum of |set| / G).
  double get_equivalent_sweeps() const noexcept { return equivalent_sweeps_; }
  /// @return Number of lagged face-trace blocks in the GMRES unknown.
  int64_t num_delayed_blocks() const noexcept
  {
    return static_cast<int64_t>(delayed_.size());
  }
  bool is_converged() const noexcept { return converged_; }
  /// @return SweepConfig::certify: scalar-flux change (relative L2) of the
  /// certifying exact transport solve, NaN without one.
  double get_certification_error() const noexcept
  {
    return certification_error_;
  }
  /// @return SweepConfig::certify: change in k of the certifying solve
  /// (k-eigenvalue only), NaN otherwise.
  double get_certification_k_change() const noexcept
  {
    return certification_k_change_;
  }
  double get_certification_seconds() const noexcept
  {
    return certification_seconds_;
  }
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
  /// @return Bytes held (or, for GMRES, allocated during a solve) by the
  /// solver's large buffers on this rank: angular flux, face traces (current
  /// and previous), ghost traces, moments, factorization cache, and the
  /// GMRES basis of the largest group set ((restart + 4) Krylov vectors).
  std::map<std::string, int64_t> memory_report() const;

  /// @return Bytes held by the factorization cache (0 if disabled).
  int64_t factorization_bytes() const
  {
    const auto bytes = memory_report();
    const auto it = bytes.find("factorizations");
    return it == bytes.end() ? 0 : it->second;
  }
  /// @return Number of local units solved by the Kronecker-sum solver.
  int64_t num_kronecker_units() const noexcept
  {
    return std::count_if(patches_.begin(), patches_.end(),
      [](const PatchState& P) { return P.kronecker != nullptr; });
  }
  /// @return Number of local units solved by the extruded-patch solver.
  int64_t num_extruded_units() const noexcept
  {
    return std::count_if(patches_.begin(), patches_.end(),
      [](const PatchState& P) { return P.extruded != nullptr; });
  }
  /// @return Local solves, GMRES operator applications and solves that fell
  /// back to LU of the ITERATIVE solver (all patches).
  std::tuple<int64_t, int64_t, int64_t> iterative_statistics() const;
  /// @return Chunk id of each local unit (in local order).
  const std::vector<int64_t>& get_chunks() const noexcept { return chunk_of_; }
  /// @return Tasks of one sweep over every group set.
  int64_t num_tasks() const noexcept;
  const typename Mesh::Ptr& get_mesh() const noexcept { return mesh_; }
};

/// @brief Typed front end of SweepSolver for an IGA mesh of dimension
/// NumDim: the factory the bindings construct (one Python class per
/// dimension).
template<typename BlockType, int64_t NumDim>
class SweepDriver : public SweepSolver {
  static_assert(std::is_same_v<BlockType, cad::Patch>,
    "SweepDriver sweeps IGA patches (cad::Patch)");
  static_assert(NumDim >= 1 && NumDim <= 3, "NumDim must be 1, 2 or 3");

public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<SweepDriver>;

  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new SweepDriver(std::forward<Args>(args)...));
  }

protected:
  // =================================================================
  // Protected constructors
  SweepDriver(typename Mesh::Ptr mesh, xs::Server::Ptr xs_server,
    const parallel::ParallelContext& mpi_context, bool serial = false)
    : SweepSolver(
        NumDim, std::move(mesh), std::move(xs_server), mpi_context, serial)
  {}
};

} // namespace ttnte::sweep
