// SweepSolver: Construction, setup, partitioning and the sweep task graphs.
#include "ttnte/sweep/sweep_driver.hpp"

namespace ttnte::sweep {

namespace {

// Defaults; CMake overrides them (TTNTE_AXIS_ALIGNED_TOL,
// TTNTE_REFLECTION_MATCH_TOL)
#ifndef TTNTE_AXIS_ALIGNED_TOL
#define TTNTE_AXIS_ALIGNED_TOL 1e-6
#endif
#ifndef TTNTE_REFLECTION_MATCH_TOL
#define TTNTE_REFLECTION_MATCH_TOL 1e-12
#endif

/// Largest deviation of a reflective face's unit normals from its axis.
constexpr double AXIS_ALIGNED_TOL = TTNTE_AXIS_ALIGNED_TOL;
/// Largest 1 - cos between a reflected direction and its quadrature match.
constexpr double REFLECTION_MATCH_TOL = TTNTE_REFLECTION_MATCH_TOL;

} // namespace

torch::Tensor SweepSolver::build_trace_map(size_t dim,
  const mesh::NeighborInfo& conn, const PatchDiscretization& nb) const
{
  const auto& nb_face = nb.face(conn.dim, conn.is_upper);
  if (ndim_ == 1) {
    return torch::zeros({1}, torch::kInt64);
  }
  const auto& nb_shape = nb.shape();
  auto grid = torch::arange(nb.num_dofs(), torch::kInt64).reshape(nb_shape);
  auto face =
    grid.narrow(conn.dim, conn.is_upper ? nb_shape[conn.dim] - 1 : 0, 1);
  std::vector<int64_t> perm(conn.mapping.perm.begin(), conn.mapping.perm.end());
  bool needs_perm = false;
  for (size_t k = 0; k < perm.size(); k++)
    needs_perm |= perm[k] != static_cast<int64_t>(k);
  if (needs_perm)
    face = face.permute(perm);
  std::vector<int64_t> flip_dims;
  size_t perp = 0;
  for (int64_t d = 0; d < ndim_; d++) {
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

torch::Tensor SweepSolver::build_reflection(const FaceGeometry& face) const
{
  auto mean = (face.normal * face.weights.unsqueeze(1)).sum(0);
  int64_t axis = mean.abs().argmax().template item<int64_t>();
  auto unit = face.normal / face.normal.norm(2, 1, true);
  auto off_axis =
    (unit.abs() - torch::one_hot(torch::tensor(axis), ndim_).to(unit))
      .abs()
      .max()
      .template item<double>();
  if (off_axis > AXIS_ALIGNED_TOL) {
    throw utils::runtime_error(
      ctx("setup"), "Reflective boundaries must be axis aligned");
  }
  auto reflected = omega3_.clone();
  reflected.select(1, axis).neg_();
  torch::Tensor mismatch, index;
  if (ndim_ == 1) {
    // Slabs carry only mu
    auto dist =
      (reflected.select(1, 0).unsqueeze(1) - omega3_.select(1, 0).unsqueeze(0))
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
  if (mismatch.max().template item<double>() > REFLECTION_MATCH_TOL) {
    throw utils::runtime_error(ctx("setup"),
      "The angular quadrature is not symmetric about a reflective face");
  }
  return index;
}

std::vector<int64_t> SweepSolver::initial_directions() const
{
  const int64_t Nd = omega_.size(0);
  auto o = omega_.contiguous();
  const double* w = o.template data_ptr<double>();
  std::vector<int64_t> out;
  if (ndim_ == 1) {
    int64_t lo = 0, hi = 0, pos = -1, neg = -1;
    for (int64_t d = 0; d < Nd; d++) {
      if (w[d] < w[lo])
        lo = d;
      if (w[d] > w[hi])
        hi = d;
      if (w[d] > 0 && (pos < 0 || w[d] < w[pos]))
        pos = d;
      if (w[d] < 0 && (neg < 0 || w[d] > w[neg]))
        neg = d;
    }
    for (int64_t d : {lo, hi, pos, neg})
      if (d >= 0 && std::find(out.begin(), out.end(), d) == out.end())
        out.push_back(d);
    return out;
  }
  auto o3 = omega3_.contiguous();
  const double* w3 = o3.template data_ptr<double>();
  std::map<int64_t, std::pair<double, int64_t>> best;
  for (int64_t d = 0; d < Nd; d++) {
    int64_t pattern = 0;
    for (int64_t i = 0; i < ndim_; i++)
      pattern = 2 * pattern + (w[d * ndim_ + i] > 0 ? 1 : 0);
    double lo = 1e300, hi = 0.0;
    for (int64_t i = 0; i < 3; i++) {
      lo = std::min(lo, std::abs(w3[d * 3 + i]));
      hi = std::max(hi, std::abs(w3[d * 3 + i]));
    }
    auto it = best.find(pattern);
    if (it == best.end() || hi - lo < it->second.first)
      best[pattern] = {hi - lo, d};
  }
  for (const auto& [pattern, entry] : best)
    out.push_back(entry.second);
  return out;
}

void SweepSolver::build_crosses()
{
  crosses_.clear();
  if (config_.transport != TransportSolve::CROSS)
    return;
  // TODO: exchange tasks and reductions for multi-rank crosses
  if (comm_.size() > 1)
    throw utils::runtime_error(
      ctx("setup"), "CROSS transport solves are serial in space for now");
  if (config_.cross_inner_factor <= 0.0)
    throw utils::runtime_error(
      ctx("setup"), "cross_inner_factor must be positive");
  const auto modes = angular_modes();
  cross::CrossProblem base;
  base.num_dim = ndim_;
  base.num_polar = modes[0];
  base.num_azimuthal = ndim_ == 1 ? 1 : modes[1];
  base.num_space = fiber_length_;
  base.weights = weights_;
  base.harmonics = harmonics_;
  base.initial_directions = initial_directions();
  base.patch_offsets = fiber_offsets_;
  for (const auto& P : patches_) {
    base.patch_shapes.push_back(P.disc->shape());
    if (config_.cross.weighted) {
      base.mass_bands.push_back(
        linalg::banded::symmetric_band_from_pairs(P.disc->pair_row(),
          P.disc->pair_col(), P.disc->mass_values(), P.disc->num_dofs())
          .first);
    }
  }
  // Energy weighting by lethargy when the cross sections carry their group
  // boundaries (the server checked them: G + 1, decreasing, positive)
  const auto widths = config_.cross.weighted ? xs_server_->lethargy_widths()
                                             : std::vector<double> {};
  if (!widths.empty() && static_cast<int64_t>(widths.size()) != num_groups_)
    throw utils::runtime_error(ctx("setup"),
      "The cross-section server's group boundaries do not match the groups");
  for (size_t s = 0; s < group_indices_.size(); s++) {
    auto problem = base;
    problem.num_groups = static_cast<int64_t>(group_indices_[s].size());
    // Each set's cross sees only its own groups, and so only their widths
    for (int64_t g :
      widths.empty() ? std::vector<int64_t> {} : group_indices_[s])
      problem.lethargy_widths.push_back(widths[g]);
    problem.sweep = [this, s](const std::vector<FiberBlock>& blocks,
                      double* out, const FiberDone& done) {
      auto global = blocks;
      for (auto& b : global)
        for (auto& g : b.groups)
          g = group_indices_[s][g];
      sweep_fibers(global, out, done);
    };
    crosses_.push_back(
      cross::LayoutCross::create(std::move(problem), config_.cross));
  }
}

void SweepSolver::build_range_dag()
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

void SweepSolver::reset_state(bool unit_scalar_flux)
{
  cross_tolerances_.clear();
  cross_fibers_.clear();
  cross_max_ranks_.clear();
  cross_errors_.clear();
  cross_ranks_.clear();
  cross_histories_.clear();
  cross_seconds_ = 0.0;
  cross_sweep_seconds_ = 0.0;
  cross_timings_.clear();
  solve_timings_.clear();
  rhs_ratio_.assign(group_indices_.size(), 0.0);
  anchor_x_.assign(group_indices_.size(), {});
  anchor_y_.assign(group_indices_.size(), {});
  anchor_source_.assign(group_indices_.size(), {});
  anchor_steps_.assign(group_indices_.size(), 0);
  has_anchor_.assign(group_indices_.size(), 0);
  incremental_solves_ = full_solves_ = 0;
  for (auto& P : patches_) {
    P.moments.zero_();
    if (unit_scalar_flux)
      P.moments[0].fill_(1.0);
    for (size_t f = 0; f < P.links.size(); f++) {
      for (auto* t : {&P.trace_current[f], &P.trace_previous[f],
             &P.ghost_current[f], &P.ghost_previous[f]})
        if (t->defined())
          t->zero_();
    }
  }
  errors_.clear();
  k_history_.clear();
  sweep_times_.clear();
  outer_sweeps_.clear();
  group_set_sweeps_ = 0;
  equivalent_sweeps_ = 0.0;
  converged_ = false;
  certification_error_ = std::numeric_limits<double>::quiet_NaN();
  certification_k_change_ = std::numeric_limits<double>::quiet_NaN();
  certification_seconds_ = 0.0;
}

std::vector<int> SweepSolver::partition(const std::vector<PatchState>& nodes)
{
  std::vector<int> rank(nodes.size(), comm_.rank());
  if (comm_.size() == 1)
    return rank;
  std::unordered_map<int64_t, int> gid2rank = partition_;
  if (gid2rank.empty()) {
    // Vertex weights: a patch's sweep work scales with its dofs (the
    // directions and groups are the same for every patch)

    // TODO: Allow different partitioning heuristics in the future. Some that
    // might help with load balancing better in tensor network methods.
    auto dofs =
      std::make_shared<parallel::heuristics::DofHeuristic<cad::Patch>>();
    dofs->finalize(mesh_);
    parallel::LoadBalancer<cad::Patch> balancer(comm_.size());
    gid2rank =
      balancer
        .compute_partition(mesh_->build_connectivity_graph(), comm_, {dofs})
        .second;
  }
  std::vector<int64_t> keep;
  for (size_t i = 0; i < nodes.size(); i++) {
    auto it = gid2rank.find(nodes[i].gid);
    if (it == gid2rank.end() || it->second < 0 || it->second >= comm_.size()) {
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

void SweepSolver::build_dag()
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
      direction_blocks_.push_back({a, std::vector<int64_t>(dirs.begin() + d0,
                                        dirs.begin() + std::min(d0 + per, n))});
    }
  }

  for (size_t s = 0; s < group_indices_.size(); s++) {
    auto dag = std::make_unique<task::TaskGraph>();
    for (size_t b = 0; b < direction_blocks_.size(); b++) {
      auto tasks = add_chunk_tasks(*dag, direction_blocks_[b].set,
        "sweep_b" + std::to_string(b) + "_g" + std::to_string(s),
        [this, b, s](size_t c) {
          auto t0 = std::chrono::steady_clock::now();
          run_chunk(b, s, c);
          task_nanoseconds_ +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - t0)
              .count();
        });
      add_exchange_tasks(*dag, tasks, b, s);
    }
    dags_.push_back(std::move(dag));
  }
}

std::vector<task::Task*> SweepSolver::add_chunk_tasks(task::TaskGraph& dag,
  size_t set, const std::string& label, std::function<void(size_t)> body)
{
  const auto& chunks = schedules_[set];
  std::vector<task::Task*> tasks(chunks.size());
  for (size_t c = 0; c < chunks.size(); c++) {
    tasks[c] = dag.create_task(
      task::DeviceTarget::CPU_ASYNC, label + "_c" + std::to_string(c));
    tasks[c]->set_payload([body, c]() {
      body(c);
      return task::TaskStatus::COMPLETED;
    });
  }
  for (size_t c = 0; c < chunks.size(); c++)
    for (int64_t u : chunks[c].upstream)
      tasks[c]->add_dependency(tasks[u]);
  return tasks;
}

void SweepSolver::add_exchange_tasks(task::TaskGraph& dag,
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
  // MPI counts are int: one message is a block's directions x the set's
  // groups x the face's dofs
  auto message_count = [&](int64_t Nf) {
    const int64_t count = nd * ng * Nf;
    if (count > std::numeric_limits<int>::max()) {
      throw utils::runtime_error(
        ctx("setup"), "An interface message exceeds the MPI count limit (" +
                        std::to_string(count) +
                        " values); lower directions_per_task or "
                        "group_set_size");
    }
    return static_cast<int>(count);
  };

  for (const auto& ex : recvs_[a]) {
    const int64_t Nf = patches_[ex.local].links[ex.face].neighbor_face_dofs;
    exchange_buffers_.emplace_back(nd * ng * Nf);
    double* buf = exchange_buffers_.back().data();
    const int count = message_count(Nf);
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
    const int count = message_count(Nf);
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

SweepSolver::SweepSolver(int64_t num_dim, Mesh::Ptr mesh,
  xs::Server::Ptr xs_server, const parallel::ParallelContext& mpi_context,
  bool serial)
  : ndim_(num_dim), mesh_(std::move(mesh)), xs_server_(std::move(xs_server)),
    comm_(serial ? parallel::Communicator::self()
                 : parallel::Communicator::world()),
    serial_(serial)
{
  (void)mpi_context;
  if (ndim_ < 1 || ndim_ > 3)
    throw utils::runtime_error(ctx("SweepSolver"), "num_dim must be 1, 2 or 3");
  if (!mesh_->is_finalized())
    mesh_->finalize();
  if (!xs_server_->is_finalized())
    xs_server_->finalize();
}

void SweepSolver::setup(const math::QuadratureSet::Ptr& qset,
  const SweepConfig& config, const std::unordered_map<int64_t, int64_t>& chunks)
{
  qset_ = qset;
  config_ = config;
  if (config_.inner_solver == InnerSolver::RICHARDSON &&
      (config_.correction_form || config_.incremental_transport)) {
    throw utils::runtime_error(ctx("setup"),
      "correction_form and incremental_transport need inner_solver = GMRES");
  }
  has_fission_ = false;
  has_external_ = false;
  patches_.clear();
  gid_to_local_.clear();

  const bool distributed = comm_.size() > 1;
  if (distributed &&
      mesh_->get_blocks().size() != mesh_->get_global_num_blocks()) {
    throw utils::runtime_error(ctx("setup"),
      "Multi-rank sweeps need the full (undistributed) mesh on every rank: "
      "setup() builds the global sweep graph, then partitions the mesh");
  }

  std::tie(omega_, omega3_, weights_) = discrete_ordinates(*qset_, ndim_);
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
      P.scatter = mat.get_scatter_gtg().to(torch::kFloat64).cpu().contiguous();
      num_orders = std::max<int64_t>(num_orders, P.scatter.size(0));
    }
    has_fission_ = has_fission_ || mat.is_fissile();
    if (mat.is_fissile()) {
      P.fission = torch::outer(mat.get_chi().to(torch::kFloat64).cpu(),
        mat.get_nu_fission().to(torch::kFloat64).cpu())
                    .unsqueeze(0)
                    .contiguous();
    }
    if (!nodes.empty() && P.sigma_t.size(0) != num_groups_)
      throw utils::runtime_error(
        ctx("setup"), "All materials must have the same number of groups");
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
      has_external_ = true;
    }
    gid_to_node[P.gid] = nodes.size();
    nodes.push_back(std::move(P));
  }

  std::tie(harmonics_, moment_order_) =
    discrete_harmonics(*qset_, ndim_, num_orders);
  weighted_harmonics_ = (harmonics_ * weights_.unsqueeze(1)).contiguous();
  moment_order_raw_.assign(moment_order_.template data_ptr<int64_t>(),
    moment_order_.template data_ptr<int64_t>() + moment_order_.numel());
  harmonics_ = harmonics_.contiguous();
  omega_ = omega_.contiguous();

  // Non-conforming interfaces: both faces integrate with one shared
  // quadrature (union of the knot spans, the larger degree), the second
  // face with the exact negative of the first's normal, so the outflow of
  // one side is the inflow of the other
  std::map<std::pair<size_t, int64_t>, std::array<torch::Tensor, 4>> couplings;
  for (size_t i = 0; i < nodes.size() && ndim_ > 1; i++) {
    auto& P = nodes[i];
    for (int64_t dim = 0; dim < ndim_; dim++) {
      for (bool is_upper : {false, true}) {
        const auto& info = P.block->get_boundary_info(dim, is_upper);
        if (info.get_type() != physics::BoundaryType::INTERNAL ||
            info.get_connections().size() != 1)
          continue;
        const auto& conn = info.get_connections()[0];
        if (conn.conforming || P.gid > conn.gid)
          continue;
        const size_t j = gid_to_node.at(conn.gid);
        auto& Q = nodes[j];
        auto iq = cad::interface_quadrature(
          *P.block, dim, *Q.block, conn.dim, conn.mapping);
        cad::check_shared_parametrization(*P.block->get_boundary(dim, is_upper),
          *Q.block->get_boundary(conn.dim, conn.is_upper), iq);
        auto fa = P.disc->evaluate_face(dim, is_upper, iq.u_first);
        auto fb = Q.disc->evaluate_face(conn.dim, conn.is_upper, iq.u_second);
        if (((fa.normal * fb.normal).sum(1) > 0).any().template item<bool>()) {
          throw utils::runtime_error(ctx("setup"),
            "The normals of a non-conforming interface are not opposite");
        }
        P.disc->set_face_quadrature(
          dim, is_upper, iq.u_first, iq.weights, fa.normal);
        Q.disc->set_face_quadrature(
          conn.dim, conn.is_upper, iq.u_second, iq.weights, -fa.normal);
        auto pairs = [](const FacePoints& own, const FacePoints& src) {
          const int64_t q = own.R.size(0), na = own.R.size(1),
                        nb = src.R.size(1);
          return std::array<torch::Tensor, 4> {torch::arange(q, torch::kInt64)
                                                 .view({q, 1, 1})
                                                 .expand({q, na, nb})
                                                 .reshape({-1})
                                                 .contiguous(),
            own.index.unsqueeze(2)
              .expand({q, na, nb})
              .reshape({-1})
              .contiguous(),
            src.index.unsqueeze(1)
              .expand({q, na, nb})
              .reshape({-1})
              .contiguous(),
            (own.R.unsqueeze(2) * src.R.unsqueeze(1))
              .reshape({-1})
              .contiguous()};
        };
        couplings[{i, 2 * dim + (is_upper ? 1 : 0)}] = pairs(fa, fb);
        couplings[{j, 2 * static_cast<int64_t>(conn.dim) +
                        (conn.is_upper ? 1 : 0)}] = pairs(fb, fa);
      }
    }
  }

  // Face links and interfaces between nodes
  std::vector<SweepInterface> interfaces;
  std::map<std::pair<int64_t, int64_t>, int64_t> owner_face_to_interface;
  for (size_t i = 0; i < nodes.size(); i++) {
    auto& P = nodes[i];
    P.links.resize(2 * ndim_);
    for (int64_t dim = 0; dim < ndim_; dim++) {
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
              "face (non-conforming faces are split per neighbor)");
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
          auto coupling = couplings.find({i, f});
          if (coupling != couplings.end()) {
            link.coupling_q = coupling->second[0];
            link.coupling_a = coupling->second[1];
            link.coupling_b = coupling->second[2];
            link.coupling_val = coupling->second[3];
          } else {
            link.trace_map =
              build_trace_map(dim, conn, *nodes[link.neighbor].disc)
                .contiguous();
          }
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
  graph_ = SweepGraph::create(
    static_cast<int64_t>(nodes.size()), std::move(interfaces), num_directions);

  // Partition, keep this rank's patches and turn links to other ranks'
  // patches into ghost links
  node_rank_ = partition(nodes);
  node_to_local_.assign(nodes.size(), -1);
  for (size_t node = 0; node < nodes.size(); node++) {
    if (node_rank_[node] != comm_.rank())
      continue;
    node_to_local_[node] = static_cast<int64_t>(patches_.size());
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
  const int64_t gs = config_.group_set_size > 0
                       ? std::min(config_.group_set_size, num_groups_)
                       : num_groups_;
  group_indices_.clear();
  group_masks_.clear();
  group_complements_.clear();
  for (int64_t g0 = 0; g0 < num_groups_; g0 += gs) {
    group_indices_.emplace_back(std::min(g0 + gs, num_groups_) - g0);
    std::iota(group_indices_.back().begin(), group_indices_.back().end(), g0);
    std::vector<uint8_t> mask(num_groups_, 0), complement(num_groups_, 1);
    for (int64_t k : group_indices_.back()) {
      mask[k] = 1;
      complement[k] = 0;
    }
    group_masks_.push_back(std::move(mask));
    group_complements_.push_back(std::move(complement));
  }

  // Lagged traces: reflective faces (the outflow directions, read for
  // their reflections) and interface faces read in a LAGGED state
  // Key: (unit, face, ghost) of the buffer holding the lagged trace
  std::map<std::tuple<int64_t, int64_t, bool>, std::set<int64_t>> lagged;
  for (size_t i = 0; i < patches_.size(); i++) {
    const auto& P = patches_[i];
    for (size_t f = 0; f < P.links.size(); f++) {
      const auto& link = P.links[f];
      if (link.type == physics::BoundaryType::REFLECTIVE) {
        auto& dirs = lagged[{static_cast<int64_t>(i), f, false}];
        auto outflow = P.disc->faces()[f].upwind_weights(omega_, true).sum(1);
        const double* w = outflow.template data_ptr<double>();
        for (int64_t d = 0; d < num_directions; d++)
          if (w[d] > 0.0)
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

  // Buffers; previous traces are only read where they are lagged
  std::set<std::tuple<int64_t, int64_t, bool>> lagged_faces;
  for (const auto& blk : delayed_)
    lagged_faces.insert({blk.local, blk.face, blk.ghost});
  const int64_t M = harmonics_.size(1);
  for (size_t i = 0; i < patches_.size(); i++) {
    auto& P = patches_[i];
    const int64_t N = P.disc->num_dofs();
    P.trace_current.assign(P.links.size(), torch::Tensor());
    P.trace_previous.assign(P.links.size(), torch::Tensor());
    for (size_t f = 0; f < P.links.size(); f++) {
      if (P.links[f].has_buffer) {
        const int64_t Nf = P.disc->faces()[f].num_dofs();
        P.trace_current[f] =
          torch::zeros({num_directions, num_groups_, Nf}, torch::kFloat64);
        if (lagged_faces.count({static_cast<int64_t>(i), f, false}))
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
        if (lagged_faces.count({static_cast<int64_t>(i), f, true}))
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
    const bool structured =
      config_.local_solver == LocalSolverType::SYLVESTER ||
      config_.local_solver == LocalSolverType::ITERATIVE;
    // Fully separable first, then extruded (partially separable), else the
    // iterative solver (ITERATIVE) or banded LU
    P.kronecker = structured ? KroneckerSolver::try_create(*P.disc) : nullptr;
    P.extruded = structured && !P.kronecker
                   ? ExtrudedSolver::try_create(*P.disc)
                   : nullptr;
    P.iterative = config_.local_solver == LocalSolverType::ITERATIVE &&
                      !P.kronecker && !P.extruded
                    ? IterativeSolver::create(*P.disc, config_.local_tol,
                        config_.local_restart, config_.local_max_iter)
                    : nullptr;
    if (config_.cache_factorizations && !P.kronecker && !P.extruded &&
        !P.iterative) {
      const bool banded = config_.local_solver != LocalSolverType::DENSE_LU;
      const int64_t size = banded ? N * P.disc->ldab() : N * N;
      P.factors =
        torch::empty({num_directions, num_groups_, size}, torch::kFloat64);
      P.factor_pivots =
        torch::empty({num_directions, num_groups_, N}, torch::kInt64);
    } else {
      P.factors = torch::Tensor();
      P.factor_pivots = torch::Tensor();
    }
    P.factor_ready.assign(
      P.factors.defined() ? num_directions * num_groups_ : 0, 0);
    P.psi = config_.store_angular_flux
              ? torch::zeros({num_directions, num_groups_, N}, torch::kFloat64)
              : torch::Tensor();
  }

  fiber_offsets_.assign(patches_.size(), 0);
  fiber_length_ = 0;
  for (size_t i = 0; i < patches_.size(); i++) {
    fiber_offsets_[i] = fiber_length_;
    fiber_length_ += patches_[i].disc->num_dofs();
  }

  scheduler_ = std::make_unique<task::TaskScheduler>(
    std::max<int64_t>(1, config_.num_threads));
  build_dag();
  build_range_dag();
  build_crosses();
  is_setup_ = true;

  if (config_.verbose && config_.progress_seconds >= 0 && comm_.rank() == 0) {
    const auto report = memory_report();
    int64_t total = 0;
    std::cout << "-- sweep setup: " << patches_.size() << " units, "
              << num_directions << " directions, " << num_groups_
              << " groups in " << group_indices_.size() << " group sets, "
              << delayed_.size() << " lagged trace blocks; memory (GB):";
    for (const auto& [key, b] : report) {
      if (key == "krylov_vector")
        continue;
      total += b;
      std::cout << " " << key << " " << std::fixed << std::setprecision(2)
                << b / 1e9;
    }
    std::cout << ", total " << total / 1e9 << std::defaultfloat << std::endl;
  }
}

std::vector<int64_t> SweepSolver::angular_modes() const
{
  if (ndim_ == 1)
    return {omega_.size(0)};
  const auto& pq = static_cast<const math::ProductQuadrature&>(*qset_);
  return {pq.get_quads()[0]->get_num_dofs(), pq.get_quads()[1]->get_num_dofs()};
}

} // namespace ttnte::sweep
