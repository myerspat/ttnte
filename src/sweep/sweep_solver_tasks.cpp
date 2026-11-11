// SweepSolver: Sweep tasks, fiber sweeps and the cross transport.
#include "ttnte/sweep/sweep_driver.hpp"

namespace ttnte::sweep {

namespace {

// Default; CMake overrides it (TTNTE_FIBER_TASKS_PER_THREAD)
#ifndef TTNTE_FIBER_TASKS_PER_THREAD
#define TTNTE_FIBER_TASKS_PER_THREAD 4
#endif

/// Task families a fiber request is split into, per thread.
constexpr int64_t TASKS_PER_THREAD = TTNTE_FIBER_TASKS_PER_THREAD;

} // namespace

const torch::Tensor& SweepSolver::delayed_buffer(
  const DelayedBlock& blk, bool current) const
{
  const auto& P = patches_[blk.local];
  if (blk.ghost)
    return current ? P.ghost_current[blk.face] : P.ghost_previous[blk.face];
  return current ? P.trace_current[blk.face] : P.trace_previous[blk.face];
}

FaceInflow SweepSolver::face_inflow(
  const AngleSet& set, size_t local, size_t f) const
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
    if (link.coupling_q.defined()) {
      in.num_coupling_pairs = link.coupling_q.numel();
      in.coupling_q = link.coupling_q.template data_ptr<int64_t>();
      in.coupling_a = link.coupling_a.template data_ptr<int64_t>();
      in.coupling_b = link.coupling_b.template data_ptr<int64_t>();
      in.coupling_val = link.coupling_val.template data_ptr<double>();
    } else {
      in.trace_map = link.trace_map.template data_ptr<int64_t>();
    }
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

SweepBlock SweepSolver::make_block(const std::vector<int64_t>& directions,
  const std::vector<int64_t>& groups) const
{
  SweepBlock block;
  block.directions = directions.data();
  block.num_block_directions = static_cast<int64_t>(directions.size());
  block.groups = groups.data();
  block.num_block_groups = static_cast<int64_t>(groups.size());
  block.num_directions = omega_.size(0);
  block.num_groups = num_groups_;
  block.num_moments = harmonics_.size(1);
  block.omega = omega_.template data_ptr<double>();
  block.harmonics = harmonics_.template data_ptr<double>();
  block.weighted_harmonics = weighted_harmonics_.template data_ptr<double>();
  block.solver = config_.local_solver;
  return block;
}

torch::Tensor SweepSolver::cross_patch_psi(size_t s, size_t local) const
{
  std::vector<int64_t> dirs(omega_.size(0)), groups(group_indices_[s].size()),
    space(patches_[local].disc->num_dofs());
  std::iota(dirs.begin(), dirs.end(), 0);
  std::iota(groups.begin(), groups.end(), 0);
  std::iota(space.begin(), space.end(), fiber_offsets_[local]);
  return crosses_[s]->evaluate(dirs, groups, space);
}

UnitSweep SweepSolver::make_unit(const AngleSet& set, int64_t local)
{
  auto& P = patches_[local];
  UnitSweep unit;
  unit.disc = P.disc.get();
  unit.kronecker = P.kronecker.get();
  unit.extruded = P.extruded.get();
  unit.iterative = P.iterative.get();
  unit.sigma_t = P.sigma_t.template data_ptr<double>();
  unit.source_moments = P.source_moments.template data_ptr<double>();
  for (size_t f = 0; f < P.links.size(); f++) {
    unit.inflow[f] = face_inflow(set, local, f);
    unit.trace_out[f] = P.links[f].has_buffer
                          ? P.trace_current[f].template data_ptr<double>()
                          : nullptr;
  }
  return unit;
}

void SweepSolver::use_factor_cache(PatchState& P, UnitSweep& unit)
{
  if (!P.factors.defined())
    return;
  unit.factors = P.factors.template data_ptr<double>();
  unit.factor_pivots = P.factor_pivots.template data_ptr<int64_t>();
  unit.factor_ready = P.factor_ready.data();
}

void SweepSolver::run_chunk(size_t block_index, size_t group_set, size_t c)
{
  thread_local std::vector<double> moments;

  const auto& blk = direction_blocks_[block_index];
  const auto& set = graph_->angle_sets()[blk.set];
  const auto& groups = group_indices_[group_set];
  const int64_t M = harmonics_.size(1);
  const int64_t ng = static_cast<int64_t>(groups.size());
  const auto block = make_block(blk.directions, groups);

  for (int64_t local : schedules_[blk.set][c].units) {
    auto& P = patches_[local];
    const int64_t N = P.disc->num_dofs();
    auto unit = make_unit(set, local);
    unit.psi = P.psi.defined() ? P.psi.template data_ptr<double>() : nullptr;
    use_factor_cache(P, unit);
    moments.assign(static_cast<size_t>(M * ng * N), 0.0);
    unit.moments = moments.data();

    sweep_unit(unit, block, workspace());

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

void SweepSolver::run_fiber_chunk(size_t t, size_t c, double* out)
{
  const auto& task = fiber_tasks_[t];
  const auto& set = graph_->angle_sets()[task.set];
  const auto block = make_block(task.directions, task.groups);

  for (int64_t local : schedules_[task.set][c].units) {
    auto& P = patches_[local];
    auto unit = make_unit(set, local);
    use_factor_cache(P, unit);
    unit.fibers = out + fiber_offsets_[local];
    unit.fiber_slot = task.slots.data();
    unit.fiber_stride = fiber_length_;
    sweep_unit(unit, block, workspace());
  }
}

void SweepSolver::cross_group_set(size_t s)
{
  // The cross's stand-in for one transport sweep of group set s: build a
  // low-rank psi of the set's groups from sampled fibers (the sources and
  // traces were set by the caller), then produce from it everything an
  // exact sweep would leave behind: the moments, the lagged traces and,
  // if stored, psi. Groups are local to the set inside the cross (0..ng-1)
  // and global (groups[bg]) in the solver's buffers.
  auto& cr = *crosses_[s];
  const auto& groups = group_indices_[s];
  const int64_t ng = static_cast<int64_t>(groups.size());
  const int64_t G = num_groups_;

  // Tolerance: cross_tol_ when the within-set solver set one for this
  // transport solve (relaxed inexact-Krylov products with cross_relax, the
  // correction form's tol_min), else the fixed CrossConfig::tol
  cr.set_tolerance(cross_tol_ > 0 ? cross_tol_ : config_.cross.tol);
  // The cross's linear algebra runs on at most cross_threads threads (its
  // matrices are small; more threads cost more in synchronization). The
  // fiber sweeps it requests still use the solver's task pool
  const int64_t solver_threads = std::max<int64_t>(1, config_.num_threads);
  std::optional<ThreadCountGuard> threads(std::in_place,
    config_.cross_threads > 0 ? std::min(config_.cross_threads, solver_threads)
                              : solver_threads);
  // Forward/backward passes until the hold-out check accepts; every fiber
  // it needs is swept through the sweep callback (sweep_fibers)
  cr.run();

  auto t0 = std::chrono::steady_clock::now();
  auto lap = [&](const char* phase) {
    const auto now = std::chrono::steady_clock::now();
    cross_timings_[phase] += std::chrono::duration<double>(now - t0).count();
    t0 = now;
  };
  // 1. Moments: (M, ng, fiber_length) from the TT (quadrature times
  // harmonics contracted with the angle cores). Scatter each patch's
  // segment of the fiber into its global groups of moments_next, which the
  // within-set iteration reads as this transport solve's result
  auto moments = cr.moments();
  const int64_t M = moments.size(0);
  const double* mom = moments.template data_ptr<double>();
  std::vector<int64_t> local_groups(ng);
  std::iota(local_groups.begin(), local_groups.end(), 0);
  for (size_t i = 0; i < patches_.size(); i++) {
    auto& P = patches_[i];
    const int64_t N = P.disc->num_dofs();
    double* dst = P.moments_next.template data_ptr<double>();
    for (int64_t m = 0; m < M; m++)
      for (int64_t bg = 0; bg < ng; bg++) {
        const double* src =
          mom + (m * ng + bg) * fiber_length_ + fiber_offsets_[i];
        std::copy(src, src + N, dst + (m * G + groups[bg]) * N);
      }
  }
  lap("moments");

  // 2. Lagged traces. The next transport solve reads every lagged edge
  // (cycle breaks, reflective faces) from the trace buffers at all of the
  // edge's directions, but the fiber sweeps only wrote the pairs the cross
  // sampled. So each delayed block (unit, face, directions) is evaluated
  // from the TT at the face's dofs and written into its current buffer,
  // at the set's groups only (other sets' traces are left alone).
  // One lagged block per task (each writes its own directions' rows); the
  // contractions inside run serially, nested in the parallel region
  at::parallel_for(
    0, static_cast<int64_t>(delayed_.size()), 1, [&](int64_t b0, int64_t b1) {
      for (int64_t b = b0; b < b1; b++) {
        const auto& blk = delayed_[b];
        const auto& face = patches_[blk.local].disc->faces()[blk.face];
        const int64_t Nf = face.num_dofs();
        auto dofs = (face.dofs + fiber_offsets_[blk.local]).contiguous();
        std::vector<int64_t> space(dofs.template data_ptr<int64_t>(),
          dofs.template data_ptr<int64_t>() + Nf);
        auto values = cr.evaluate(blk.directions, local_groups, space);
        const double* v = values.template data_ptr<double>();
        double* buf = delayed_buffer(blk, true).template data_ptr<double>();
        for (size_t i = 0; i < blk.directions.size(); i++)
          for (int64_t bg = 0; bg < ng; bg++)
            std::copy(v + (i * ng + bg) * Nf, v + (i * ng + bg + 1) * Nf,
              buf + (blk.directions[i] * G + groups[bg]) * Nf);
      }
    });
  lap("traces");

  // 3. Angular flux, if stored: each patch's psi at the set's groups,
  // evaluated from the TT at every direction and dof
  if (config_.store_angular_flux) {
    const auto group_index = torch::tensor(groups, torch::kInt64);
    for (size_t i = 0; i < patches_.size(); i++)
      patches_[i].psi.index_copy_(1, group_index, cross_patch_psi(s, i));
    lap("angular_flux");
  }
  // Restore torch's thread count before the bookkeeping
  threads.reset();

  // Per-transport-solve records (rank, fibers, hold-out error, history).
  // A full sweep of this set would be Nd * ng fibers, so the solve counts
  // fibers / (Nd * G) sweep equivalents of the whole problem
  const auto ranks = cr.ranks();
  cross_tolerances_.push_back(cr.config().tol);
  cross_fibers_.push_back(cr.num_fibers());
  cross_max_ranks_.push_back(*std::max_element(ranks.begin(), ranks.end()));
  cross_errors_.push_back(cr.holdout_error());
  cross_ranks_.push_back(ranks);
  cross_histories_.push_back(cr.history());
  cross_seconds_ += cr.seconds();
  cross_sweep_seconds_ += cr.sweep_seconds();
  group_set_sweeps_++;
  equivalent_sweeps_ += static_cast<double>(cr.num_fibers()) /
                        static_cast<double>(omega_.size(0) * G);
}

void SweepSolver::transport_group_set(size_t s)
{
  if (config_.transport == TransportSolve::CROSS)
    cross_group_set(s);
  else
    sweep_group_set(s);
}

void SweepSolver::parallel_for(
  int64_t n, const std::function<void(int64_t)>& fn)
{
  if (!range_dag_ || n < 2) {
    for (int64_t i = 0; i < n; i++)
      fn(i);
    return;
  }
  range_fn_ = &fn;
  range_size_ = n;
  // Cleared on the way out, also when a task throws
  struct Clear {
    const std::function<void(int64_t)>*& fn;
    ~Clear() { fn = nullptr; }
  } clear {range_fn_};
  range_dag_->reset();
  scheduler_->execute(*range_dag_, task::ExecutionMode::PUSH);
}

void SweepSolver::sweep_group_set(size_t s)
{
  // The exact transport solve of group set s: one full sweep of every
  // direction at the set's groups, against the source moments and lagged
  // traces the caller set. It leaves the set's new moments in moments_next
  // and its outgoing traces in trace_current (psi too, if stored); the
  // caller decides whether to commit them (commit_group_set).
  const auto& groups = group_indices_[s];
  const int64_t M = harmonics_.size(1);

  // The sweep tasks ADD their moment contributions (direction blocks of
  // every angle set reduce into the same patch under its mutex), so zero
  // the set's slice of moments_next first. Other sets' groups are left
  // as they are
  for_each_patch([&](PatchState& P) {
    const int64_t N = P.disc->num_dofs();
    double* m = P.moments_next.template data_ptr<double>();
    for (int64_t k = 0; k < M; k++)
      for (int64_t g : groups)
        std::fill(m + (k * num_groups_ + g) * N,
          m + (k * num_groups_ + g + 1) * N, 0.0);
  });

  // Run the set's DAG, built once in setup: one copy of each angle set's
  // chunk schedule per direction block (run_chunk), plus the MPI trace
  // exchange tasks. reset() puts every task back to WAITING so the same
  // graph runs again
  dags_[s]->reset();
  auto t0 = std::chrono::steady_clock::now();
  scheduler_->execute(*dags_[s], task::ExecutionMode::PUSH);
  dag_nanoseconds_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now() - t0)
                        .count();

  // One transport solve, |set| / G sweeps of the whole problem
  group_set_sweeps_++;
  equivalent_sweeps_ +=
    static_cast<double>(groups.size()) / static_cast<double>(num_groups_);
}

void SweepSolver::sweep_fibers(
  const std::vector<FiberBlock>& blocks, double* out, const FiberDone& done)
{
  // Fills one row of `out` per requested (direction, group) pair: the
  // angular flux of that pair over the whole local space (the patches'
  // dofs end to end at fiber_offsets_), swept against the current sources
  // and traces. Rows are in request order: block by block, direction-major,
  // then group. Only the requested pairs are swept and no moments or psi
  // are written, so the cross gets exact samples without a full sweep.
  if (!is_setup_)
    throw utils::runtime_error(ctx("sweep_fibers"), "Call setup() first");
  // TODO: exchange tasks for multi-rank fiber sweeps
  if (comm_.size() > 1)
    throw utils::runtime_error(
      ctx("sweep_fibers"), "Fiber sweeps are serial in space for now");
  const auto& direction_set = graph_->direction_set();
  const int64_t* set_of = direction_set.template data_ptr<int64_t>();
  const int64_t num_sets = static_cast<int64_t>(graph_->angle_sets().size());
  const int64_t per = config_.directions_per_task;

  // A run is a slice of one request block's directions that share an angle
  // set (and so one upwind schedule), at most directions_per_task long.
  // rows[i] is the output row of directions[i] at the block's first group;
  // group bg of that direction is row rows[i] + bg
  struct DirectionRun {
    size_t block;
    size_t set;
    std::vector<int64_t> directions;
    std::vector<int64_t> rows;
  };
  std::vector<DirectionRun> runs;
  // Flags of the pairs seen so far, to reject repeats
  std::vector<uint8_t> requested(omega_.size(0) * num_groups_, 0);
  // First output row of the current block (the total fiber count at the end)
  int64_t offset = 0;

  // 1. Split each block's directions by angle set, then into runs
  for (size_t k = 0; k < blocks.size(); k++) {
    const auto& blk = blocks[k];
    const int64_t ng = static_cast<int64_t>(blk.groups.size());
    // Bucket the block's directions (and their first rows) by angle set
    std::vector<std::vector<int64_t>> by_set(num_sets), rows(num_sets);
    for (size_t i = 0; i < blk.directions.size(); i++) {
      const int64_t d = blk.directions[i];
      if (d < 0 || d >= omega_.size(0))
        throw utils::runtime_error(
          ctx("sweep_fibers"), "Direction out of range");
      by_set[set_of[d]].push_back(d);
      rows[set_of[d]].push_back(offset + static_cast<int64_t>(i) * ng);
    }
    // Cut each set's bucket into runs of at most directions_per_task
    // (<= 0: the whole bucket)
    for (int64_t a = 0; a < num_sets; a++) {
      const int64_t n = static_cast<int64_t>(by_set[a].size());
      const int64_t step = per > 0 ? per : std::max<int64_t>(n, 1);
      for (int64_t i0 = 0; i0 < n; i0 += step) {
        const int64_t i1 = std::min(i0 + step, n);
        runs.push_back({k, static_cast<size_t>(a),
          {by_set[a].begin() + i0, by_set[a].begin() + i1},
          {rows[a].begin() + i0, rows[a].begin() + i1}});
      }
    }
    for (int64_t g : blk.groups)
      if (g < 0 || g >= num_groups_)
        throw utils::runtime_error(ctx("sweep_fibers"), "Group out of range");
    // A repeated pair would race on its traces and factor-cache slot
    for (int64_t d : blk.directions)
      for (int64_t g : blk.groups) {
        auto& flag = requested[d * num_groups_ + g];
        if (flag)
          throw utils::runtime_error(ctx("sweep_fibers"),
            "Pair (direction " + std::to_string(d) + ", group " +
              std::to_string(g) + ") requested twice");
        flag = 1;
      }
    offset += static_cast<int64_t>(blk.directions.size()) * ng;
  }
  // 2. Split each run's groups into pieces so the request exposes about
  // TASKS_PER_THREAD task families per thread (a cross request can be a
  // few directions, too few families by direction alone). Each piece is a
  // FiberTask; slots[i] is the row of directions[i] at the piece's first
  // group, so the kernel writes (i, bg) to row slots[i] + bg
  const int64_t threads = std::max<int64_t>(1, config_.num_threads);
  // Pieces per run: ceil(TASKS_PER_THREAD * threads / #runs)
  const int64_t wanted =
    (TASKS_PER_THREAD * threads + static_cast<int64_t>(runs.size()) - 1) /
    std::max<int64_t>(1, runs.size());
  fiber_tasks_.clear();
  for (const auto& run : runs) {
    const auto& groups = blocks[run.block].groups;
    const int64_t ng = static_cast<int64_t>(groups.size());
    // At least one group per piece
    const int64_t parts =
      std::clamp<int64_t>(wanted, 1, std::max<int64_t>(ng, 1));
    for (int64_t p = 0; p < parts; p++) {
      const int64_t g0 = p * ng / parts, g1 = (p + 1) * ng / parts;
      if (g0 == g1)
        continue;
      FiberTask task;
      task.set = run.set;
      task.directions = run.directions;
      task.groups.assign(groups.begin() + g0, groups.begin() + g1);
      for (int64_t row : run.rows)
        task.slots.push_back(row + g0);
      fiber_tasks_.push_back(std::move(task));
    }
  }

  // 3. One copy of the angle set's chunk schedule per fiber task (a task
  // family): one task per chunk with the schedule's upwind edges, the body
  // sweeping the family's directions and groups through that chunk's
  // units. Families share no edges, so they all run concurrently; within a
  // family a chunk waits for its upstream chunks' traces
  task::TaskGraph dag;
  for (size_t t = 0; t < fiber_tasks_.size(); t++) {
    const auto& chunks = schedules_[fiber_tasks_[t].set];
    auto tasks =
      add_chunk_tasks(dag, fiber_tasks_[t].set, "fiber_t" + std::to_string(t),
        [this, t, out](size_t c) { run_fiber_chunk(t, c, out); });
    if (done) {
      // The family's fibers are finished once its last chunks (those no
      // other chunk depends on) are: hand their rows to `done` from a task
      std::vector<uint8_t> feeds(chunks.size(), 0);
      for (size_t c = 0; c < chunks.size(); c++)
        for (int64_t u : chunks[c].upstream)
          feeds[u] = 1;
      auto* fin = dag.create_task(
        task::DeviceTarget::CPU_ASYNC, "fiber_t" + std::to_string(t) + "_done");
      fin->set_payload([this, t, &done]() {
        const auto& task = fiber_tasks_[t];
        const int64_t ng = static_cast<int64_t>(task.groups.size());
        for (int64_t slot : task.slots)
          done(slot, ng);
        return task::TaskStatus::COMPLETED;
      });
      for (size_t c = 0; c < chunks.size(); c++)
        if (!feeds[c])
          fin->add_dependency(tasks[c]);
    }
  }
  // 4. Run: the tasks are the parallelism, so torch inside them (e.g. the
  // `done` callback's linear algebra) is serial while the DAG runs
  ThreadCountGuard serial_torch(config_.num_threads > 1 ? 1 : 0);
  auto t0 = std::chrono::steady_clock::now();
  scheduler_->execute(dag, task::ExecutionMode::PUSH);
  dag_nanoseconds_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now() - t0)
                        .count();
  // Fibers swept, the cross's cost measure (sweep equivalents)
  num_fibers_ += offset;
}

void SweepSolver::sweep()
{
  if (!is_setup_)
    throw utils::runtime_error(ctx("sweep"), "Call setup() first");
  for_each_patch([](PatchState& P) { P.pass_moments.copy_(P.moments); });
  for (size_t s = 0; s < group_indices_.size(); s++) {
    compute_source(s, Within::MOMENTS, true, true);
    transport_group_set(s);
    commit_group_set(s);
  }
}

int64_t SweepSolver::num_tasks() const noexcept
{
  int64_t n = 0;
  for (const auto& dag : dags_)
    n += static_cast<int64_t>(dag->size());
  return n;
}

} // namespace ttnte::sweep
