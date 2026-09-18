#pragma once

#include "ttnte/driver/transport_solution.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/heuristics.hpp"
#include "ttnte/parallel/load_balancer.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include "ttnte/physics/dg_first_order_transport_assembler.hpp"
#include "ttnte/solvers/solver.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/label.hpp"
#include "ttnte/xs/server.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>

namespace ttnte::driver {

template<typename BlockType, int64_t NumDim>
class TransportDriver {
  friend parallel::LoadBalancer<BlockType>;

public:
  // =================================================================
  // Public types
  using Mesh = mesh::Mesh<BlockType>;
  using Label = utils::Label<TransportDriver>;
  using Ptr = std::shared_ptr<TransportDriver>;
  using Assembler = physics::DGFirstOrderTransportAssembler<BlockType, NumDim>;
  using Solution = TransportSolution<BlockType>;
  using Callback =
    std::function<void(const TransportDriver&, const solvers::Solver&)>;

  // Communication and load balancing
  using Communicator = parallel::Communicator;
  using LoadHeuristicPtr =
    std::shared_ptr<parallel::heuristics::LoadHeuristic<BlockType>>;
  using LoadBalancer = parallel::LoadBalancer<BlockType>;

private:
  // =================================================================
  // Private types
  /// Per-patch assembled data. assembler is null after clear_assemblers.
  struct PatchData {
    Assembler::Ptr assembler;
    linalg::LinearSystem::Ptr system;
  };

  // =================================================================
  // Private data
  /// Label
  Label label_;
  /// Shared pointer to the mesh
  Mesh::Ptr mesh_;
  /// Shared pointer to the cross section server
  xs::Server::Ptr xs_server_;

  // Communication and load balancing
  /// Communicator responsible for MPI communication
  Communicator comm_;
  /// Load balancer for running (Par)METIS
  LoadBalancer load_balancer_;
  // Per-patch assembled data (populated by assemble())
  std::unordered_map<int64_t, PatchData> patch_data_;
  /// The angular quadrature set passed to assemble(), retained so it
  /// survives clear_assemblers (needed by get_solution()'s TransportSolution
  /// to compute scalar flux after solve_eigenvalue()).
  math::QuadratureSet::Ptr angular_qset_;
  /// The assembler config passed to assemble(), retained (like
  /// angular_qset_) so the TransportSolution this driver produces can build
  /// fresh assemblers on demand -- e.g. compute_patch_balances()/
  /// patch_balance_table()/global_balance() when the caller doesn't already
  /// have one (say, clear_assemblers=true was used).
  physics::DGTransportAssemblerConfig config_;

  // States
  bool is_distributed_ = false;

  // Callback / outer-iteration diagnostics
  /// Optional user callback invoked once per outer iteration -- see
  /// set_callback().
  Callback callback_ = nullptr;
  /// Only invoke callback_ every callback_frequency_-th outer iteration
  /// (i % callback_frequency_ == 0) -- see set_callback().
  int callback_frequency_ = 1;
  /// The eigenvalue as of the most recent outer iteration of the most
  /// recent solve_eigenvalue() call. nullopt for solve_fixed_source() (no
  /// eigenvalue) or before the first outer iteration has run.
  std::optional<double> last_k_ = std::nullopt;
  /// Number of outer iterations the most recent solve_eigenvalue()/
  /// solve_fixed_source() call has run so far.
  int last_num_outer_iterations_ = 0;
  /// The scalar-flux-shape relative L2 error as of the most recent outer
  /// iteration of the most recent solve_eigenvalue()/solve_fixed_source()
  /// call.
  double last_outer_error_ = std::numeric_limits<double>::max();

  // =================================================================
  // Private constructors
  TransportDriver(Mesh::Ptr mesh, xs::Server::Ptr xs_server,
    const parallel::ParallelContext& mpi_context,
    std::optional<std::string> label = std::nullopt)
    : mesh_(std::move(mesh)), xs_server_(std::move(xs_server)),
      comm_(Communicator::world()), load_balancer_(mpi_context.world_size()),
      label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {
    // Finalize the mesh and server if not already finalized
    if (!mesh_->is_finalized()) {
      mesh_->finalize();
    }
    if (!xs_server_->is_finalized()) {
      xs_server_->finalize();
    }
  }

public:
  // =================================================================
  // Public methods
  /// @brief Build a TransportDriver and get the shared pointer to it.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(
      new TransportDriver<BlockType, NumDim>(std::forward<Args>(args)...));
  }

  /// @brief Assemble the linear system for each local mesh block.
  /// @param angular_qset Angular quadrature set.
  /// @param config Assembler configuration (formats, rounding parameters,
  /// etc.).
  void assemble(const math::QuadratureSet::Ptr& angular_qset,
    const physics::DGTransportAssemblerConfig& config)
  {
    patch_data_.clear();
    angular_qset_ = angular_qset;
    config_ = config;

    for (const auto& block : mesh_->get_blocks()) {
      const int64_t gid = block->get_gid();
      auto asm_ptr = Assembler::create(block, angular_qset, xs_server_, config);
      PatchData pd;
      pd.system = asm_ptr->assemble();
      pd.system->set_gid(gid);
      pd.assembler = std::move(asm_ptr);
      patch_data_.emplace(gid, std::move(pd));
    }
  }

  /// @brief Run the eigenvalue solver using the given solver.
  /// @param inner_solver Solver for each outer iteration's inner solve --
  /// e.g. a DDSolver for multi-patch domain decomposition, or a bare
  /// LocalSolver (such as AMEnSolver) for a single-patch problem.
  /// @param tol The convergence tolerance on the relative Frobenius (L2) error
  /// of the scalar flux shape (not normalized -- the natural 1/k
  /// eigenvalue rescaling is left in place, so this also reflects residual
  /// k-drift). Necessary but not sufficient on its own: the inner solver's
  /// forcing (see DDSolver::step()/AMEnSolver's eps forcing) ratchets
  /// strictly tighter as outer iterations proceed and never loosens, so a
  /// later outer iteration's inner solve can become accurate enough to make
  /// this metric read small even while k_global is still visibly drifting --
  /// see k_tol below.
  /// @param max_iter The maximum number of outer iterations.
  /// @param clear_assemblers Clear each patch's assembler after building the
  /// local systems (frees assembled operators no longer needed once the
  /// LinearSystem buffer is built).
  /// @param verbose Whether to print outer iteration progress.
  /// @param k_tol The convergence tolerance on k_global's own absolute
  /// iteration-to-iteration change, |k_i - k_{i-1}| (in k-units, so e.g.
  /// 1e-5 is 1 pcm). Checked independently of (in addition to) `tol`'s
  /// flux-shape criterion -- both must hold before the outer loop breaks.
  /// @return A TransportSolution holding the converged k-eigenvalue
  /// (get_k_eff()) and the raw angular flux per local patch (get_solution()).
  typename Solution::Ptr solve_eigenvalue(solvers::Solver::Ptr inner_solver,
    double tol = 1e-8, int max_iter = 500, bool clear_assemblers = true,
    bool verbose = true, double k_tol = 1e-5)
  {
    if (patch_data_.empty()) {
      throw utils::runtime_error(
        "ttnte::driver::TransportDriver::solve_eigenvalue",
        "No linear systems assembled. Call assemble() first.");
    }

    // Reset outer-iteration diagnostics exposed to callback_ -- a driver
    // instance may run solve_eigenvalue() more than once.
    last_k_ = std::nullopt;
    last_num_outer_iterations_ = 0;
    last_outer_error_ = std::numeric_limits<double>::max();

    // Initialize the solver
    double k_global = init_solver(inner_solver, clear_assemblers);
    // Previous iteration's k_global, for the independent k-convergence check
    // below -- seeded from the pre-loop value so the very first outer
    // iteration's k_error reflects the initial guess's own drift, not a
    // spurious zero.
    double k_prev = k_global;

    // Ensure all worker threads have set their CUDA device before the first
    // iteration so cuBLAS context initialization does not produce warnings.
    inner_solver->wait_for_thread_init();

    parallel::Request ereq;
    parallel::Request kreq;
    const double one = 1.0;
    const auto& local_systems = inner_solver->get_local_systems();
    verbose = verbose && comm_.rank() == 0;

    double error = std::numeric_limits<double>::max();
    double k_error = std::numeric_limits<double>::max();
    inner_solver->update_convergence_criteria(error);

    for (const auto& sys : local_systems) {
      const auto& x = sys->get_state();
      std::cout << "GID: " << sys->get_gid()
                << ", Discretization: " << x.as_tt().get_m_modes() << std::endl;
    }

    // Temporarily move the angular quadrature set (deliberately often
    // CPU-resident, since it's tiny) to match the local systems' own
    // device/dtype for the duration of the solve, since
    // angular_qset_->integrate() is now called every outer iteration below
    // for the scalar-flux-shape convergence check -- this avoids
    // re-transferring the small weight tensors on every single call
    // (QuadratureSet::integrate()'s own per-call device guard then becomes
    // a no-op). Restored to its original device/dtype once the solve
    // completes, even if it throws.
    const torch::Device original_qset_device = angular_qset_->get_device();
    const torch::ScalarType original_qset_dtype = angular_qset_->get_dtype();
    if (!local_systems.empty()) {
      const auto& op = local_systems[0]->get_interior_op();
      angular_qset_->to_(op.get_device(), op.get_dtype());
    }
    struct QsetRestoreGuard {
      math::QuadratureSet::Ptr qset;
      torch::Device device;
      torch::ScalarType dtype;
      ~QsetRestoreGuard() { qset->to_(device, dtype); }
    } qset_restore_guard {
      angular_qset_, original_qset_device, original_qset_dtype};

    // Previous patch scalar fluxes for computing the outer convergence
    // error. Reduced via angular_qset_->integrate() rather than diffing the
    // raw angular flux: converging the full angular flux is a stricter
    // (and, for a low-rank TT representation, somewhat misaligned)
    // criterion than what's physically needed -- k-eff, reaction rates, and
    // everything downstream only ever depend on the scalar flux (0th
    // angular moment). The eigenvalue typically converges well before the
    // scalar flux SHAPE does, making shape the binding criterion here.
    // Deliberately NOT normalized: the natural 1/k rescaling below (
    // fsrc->set_eigval(1/k_global); fsrc->scale()) is the physically
    // meaningful power-iteration update, so leaving it in means this metric
    // reflects BOTH residual k-drift and shape-drift combined -- if k
    // hasn't settled, dividing by a wrong k still causes systematic
    // amplitude drift here even once the shape itself is stable.
    std::vector<linalg::State> states(local_systems.size());

    // Initial scalar flux snapshot, before any Schwarz sweep has run. Inside
    // the loop below, each iteration's post-step() scalar flux is reused
    // directly as the next iteration's "previous" snapshot (see below) rather
    // than recomputing this same integrate() call twice per outer iteration.
    for (size_t idx = 0; idx < local_systems.size(); ++idx) {
      states[idx] = angular_qset_->integrate(local_systems[idx]->get_state(),
        inner_solver->get_eps(), inner_solver->get_max_rank());
    }

    // Begin transport iteration
    auto start = std::chrono::high_resolution_clock::now();
    int num_outer_iterations = 0;
    int total_inner_iterations = 0;
    std::vector<double> outer_k_history;
    std::vector<double> outer_k_error_history;
    std::vector<double> outer_flux_error_history;
    std::vector<int> inner_outer_iter_history;
    std::vector<double> inner_schwarz_error_history;

    for (int i = 0; i < max_iter; i++) {
      // Set before the possible break below so it always reflects the
      // number of outer iterations actually executed (0..i inclusive).
      num_outer_iterations = i + 1;

      // Set eigenvalues of the multiplying systems
      for (const auto& sys : local_systems) {
        const auto& src = sys->get_source();
        if (src && src->is_eigenvalue()) {
          auto fsrc = std::static_pointer_cast<linalg::EigenSource>(src);
          fsrc->set_eigval(one / k_global);
          fsrc->scale();
        }
      }

      // Run the DD solver
      inner_solver->step();
      total_inner_iterations += inner_solver->last_num_iterations();
      for (double inner_error : inner_solver->last_errors()) {
        inner_outer_iter_history.push_back(i + 1);
        inner_schwarz_error_history.push_back(inner_error);
      }

      // Compute the local error in the scalar flux shape using a relative
      // L2 error to the last iteration, then store this iteration's scalar
      // flux back into states[idx] so the next iteration's "previous"
      // snapshot is this one -- avoids recomputing the same integrate() call
      // twice per outer iteration.
      double local_outer_sums[2] = {0.0, 0.0};
      for (size_t idx = 0; idx < local_systems.size(); ++idx) {
        if (!states[idx].defined())
          continue;

        linalg::State scalar_flux =
          angular_qset_->integrate(local_systems[idx]->get_state(),
            inner_solver->get_eps(), inner_solver->get_max_rank());
        linalg::State diff = scalar_flux - states[idx];
        const double n_diff = diff.norm();
        const double n_prev = states[idx].norm();
        local_outer_sums[0] += n_diff * n_diff;
        local_outer_sums[1] += n_prev * n_prev;

        states[idx] = std::move(scalar_flux);
      }

      // Sum the errors across all MPI ranks
      double global_outer_sums[2] = {local_outer_sums[0], local_outer_sums[1]};
      if (comm_.size() > 1) {
        ereq = comm_.iallreduce(
          local_outer_sums, global_outer_sums, 2, parallel::MPIOp::SUM);
      }

      // Calculate the local fission source for this MPI rank
      double k = 0.0;
      for (const auto& sys : local_systems) {
        const auto& src = sys->get_source();
        if (src && src->is_eigenvalue()) {
          auto fsrc = std::static_pointer_cast<linalg::EigenSource>(src);
          fsrc->update(sys->get_state(), inner_solver->get_eps(),
            inner_solver->get_max_rank());
          k += fsrc->get_total_source();
        }
      }

      // Sum the fission sources across all MPI ranks
      if (comm_.size() > 1) {
        kreq = comm_.iallreduce(&k, &k_global, 1, parallel::MPIOp::SUM);
      } else {
        k_global = k;
      }

      // Wait for both allreduces (independent of each other -- reordered
      // here, relative to the original single kreq.wait() below the
      // update_convergence_criteria() call, so k_global is valid before
      // callback_ fires; see the comment on that call below) and update
      // outer iteration errors.
      ereq.wait();
      kreq.wait();
      error = (global_outer_sums[1] > 0.0)
                ? std::sqrt(global_outer_sums[0] / global_outer_sums[1])
                : std::numeric_limits<double>::max();

      // Update outer-iteration diagnostics exposed to callback_, and invoke
      // it (if set) BEFORE update_convergence_criteria() below -- for a bare
      // LocalSolver, that call tightens eps_ for the NEXT outer iteration,
      // so the callback must fire first to see the eps that actually
      // produced this iteration's local_systems state (same rationale as
      // DDSolver::step()'s callback_ placement).
      last_k_ = k_global;
      last_num_outer_iterations_ = num_outer_iterations;
      last_outer_error_ = error;
      if (callback_ && i % callback_frequency_ == 0) {
        callback_(*this, *inner_solver);
      }

      // Feed the outer error to the solver's generic convergence hook. This
      // is a no-op for a DDSolver (its forcing is entirely self-contained,
      // driven by its own internal partial-current error -- see
      // DDSolver::step()); for a bare LocalSolver used standalone (no domain
      // decomposition, e.g. a single-patch problem) this is the only signal
      // that ever drives its eps forcing, since nothing else calls this on
      // it.
      inner_solver->update_convergence_criteria(error);

      // Independent k-convergence check -- the flux-shape `error` above can
      // read small even while k is still drifting (see solve_eigenvalue()'s
      // doc comment on `tol`), so k_global's own absolute change is tracked
      // and checked separately rather than trusting it to be implied by the
      // flux-shape metric.
      k_error = std::abs(k_global - k_prev);
      k_prev = k_global;

      outer_k_history.push_back(k_global);
      outer_k_error_history.push_back(k_error);
      outer_flux_error_history.push_back(error);

      for (const auto& sys : local_systems) {
        const auto& x = sys->get_state();
        std::cout << "GID: " << sys->get_gid()
                  << ", Ranks: " << x.as_tt().get_ranks()
                  << ", Compression: " << x.get_compression() << std::endl;
      }

      if (verbose) {
        std::cout << "-- (" << i << "): k = " << std::fixed
                  << std::setprecision(6) << k_global
                  << ", k Error = " << std::fixed << std::setprecision(6)
                  << k_error << ", Scalar Flux L2 Error = " << std::fixed
                  << std::setprecision(10) << error
                  << ", Elapsed Time = " << std::fixed << std::setprecision(3)
                  << static_cast<double>(
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count()) *
                       1e-3
                  << " s" << std::defaultfloat << std::endl;
      }
      if (error < tol && k_error < k_tol) {
        break;
      }
    }

    // Remove the linear systems from GPU
    inner_solver->finalize();

    if (verbose) {
      std::cout << "-- "
                << ((error < tol && k_error < k_tol) ? "Converged!"
                                                     : "Failed to Converge!")
                << std::endl;
    }
    for (const auto& sys : local_systems) {
      const auto& x = sys->get_state();
      std::cout << "GID: " << sys->get_gid()
                << ", Ranks: " << x.as_tt().get_ranks()
                << ", Compression: " << x.get_compression() << std::endl;
    }

    // Build the TransportSolution holding the raw angular flux per local
    // patch; call TransportSolution::compute_scalar_flux() on the result to
    // get a spatial-only field suitable for plotting/averaging.
    auto solution = Solution::create(Communicator::world(), mesh_,
      angular_qset_, xs_server_, config_, k_global);
    solution->set_iteration_counts(
      num_outer_iterations, total_inner_iterations);
    solution->set_convergence_history(std::move(outer_k_history),
      std::move(outer_k_error_history), std::move(outer_flux_error_history),
      std::move(inner_outer_iter_history),
      std::move(inner_schwarz_error_history));
    for (const auto& sys : local_systems) {
      solution->add_local_patch(sys->get_gid(), sys->get_state());
    }
    return solution;
  }

  /// @brief Run the fixed-source solver using the given solver. Unlike
  /// solve_eigenvalue(), there is no eigenvalue to update/rescale each outer
  /// iteration -- every attached Source (volumetric and/or boundary-incident,
  /// see DGFirstOrderTransportAssembler::assemble()) is already fixed at
  /// assembly time, so this simply iterates the DD sweep to convergence.
  /// @param inner_solver Solver for each outer iteration's inner solve -- e.g.
  /// a DDSolver for multi-patch domain decomposition, or a bare LocalSolver
  /// (such as AMEnSolver) for a single-patch problem.
  /// @param tol The convergence tolerance on the relative Frobenius (L2)
  /// error of the scalar flux shape between successive outer iterations.
  /// @param max_iter The maximum number of outer iterations.
  /// @param clear_assemblers Clear each patch's assembler after building the
  /// local systems (frees assembled operators no longer needed once the
  /// LinearSystem buffer is built).
  /// @param verbose Whether to print outer iteration progress.
  /// @return A TransportSolution holding the raw angular flux per local patch
  /// (get_solution()); k_eff is left unset (nullopt) since fixed-source
  /// problems have no eigenvalue.
  typename Solution::Ptr solve_fixed_source(solvers::Solver::Ptr inner_solver,
    double tol = 1e-8, int max_iter = 500, bool clear_assemblers = true,
    bool verbose = true)
  {
    if (patch_data_.empty()) {
      throw utils::runtime_error(
        "ttnte::driver::TransportDriver::solve_fixed_source",
        "No linear systems assembled. Call assemble() first.");
    }

    // Initialize the solver. The returned total fission source is unused
    // here -- fixed-source problems have no eigenvalue to seed.
    init_solver(inner_solver, clear_assemblers);

    // Ensure all worker threads have set their CUDA device before the first
    // iteration so cuBLAS context initialization does not produce warnings.
    inner_solver->wait_for_thread_init();

    // Reset outer-iteration diagnostics exposed to callback_ -- a driver
    // instance may run solve_fixed_source() more than once. last_k_ stays
    // nullopt: fixed-source problems have no eigenvalue.
    last_k_ = std::nullopt;
    last_num_outer_iterations_ = 0;
    last_outer_error_ = std::numeric_limits<double>::max();

    parallel::Request ereq;
    const auto& local_systems = inner_solver->get_local_systems();
    verbose = verbose && comm_.rank() == 0;

    double error = std::numeric_limits<double>::max();
    inner_solver->update_convergence_criteria(error);

    // Temporarily move the angular quadrature set to match the local
    // systems' own device/dtype for the duration of the solve -- see
    // solve_eigenvalue()'s identical guard for the full rationale.
    const torch::Device original_qset_device = angular_qset_->get_device();
    const torch::ScalarType original_qset_dtype = angular_qset_->get_dtype();
    if (!local_systems.empty()) {
      const auto& op = local_systems[0]->get_interior_op();
      angular_qset_->to_(op.get_device(), op.get_dtype());
    }
    struct QsetRestoreGuard {
      math::QuadratureSet::Ptr qset;
      torch::Device device;
      torch::ScalarType dtype;
      ~QsetRestoreGuard() { qset->to_(device, dtype); }
    } qset_restore_guard {
      angular_qset_, original_qset_device, original_qset_dtype};

    // Previous patch scalar fluxes for computing the outer convergence
    // error -- see solve_eigenvalue()'s identical pattern for the rationale
    // behind reducing to the scalar flux rather than diffing the raw
    // angular flux.
    std::vector<linalg::State> states(local_systems.size());
    for (size_t idx = 0; idx < local_systems.size(); ++idx) {
      states[idx] = angular_qset_->integrate(local_systems[idx]->get_state(),
        inner_solver->get_eps(), inner_solver->get_max_rank());
    }

    // Begin transport iteration
    auto start = std::chrono::high_resolution_clock::now();
    int num_outer_iterations = 0;
    int total_inner_iterations = 0;
    std::vector<double> outer_flux_error_history;
    std::vector<int> inner_outer_iter_history;
    std::vector<double> inner_schwarz_error_history;
    for (int i = 0; i < max_iter; i++) {
      // Set before the possible break below so it always reflects the
      // number of outer iterations actually executed (0..i inclusive).
      num_outer_iterations = i + 1;

      // Run the DD solver
      inner_solver->step();
      total_inner_iterations += inner_solver->last_num_iterations();
      for (double inner_error : inner_solver->last_errors()) {
        inner_outer_iter_history.push_back(i + 1);
        inner_schwarz_error_history.push_back(inner_error);
      }

      // Compute the local error in the scalar flux shape using a relative
      // L2 error to the last iteration.
      double local_outer_sums[2] = {0.0, 0.0};
      for (size_t idx = 0; idx < local_systems.size(); ++idx) {
        if (!states[idx].defined())
          continue;

        linalg::State scalar_flux =
          angular_qset_->integrate(local_systems[idx]->get_state(),
            inner_solver->get_eps(), inner_solver->get_max_rank());
        linalg::State diff = scalar_flux - states[idx];
        const double n_diff = diff.norm();
        const double n_prev = states[idx].norm();
        local_outer_sums[0] += n_diff * n_diff;
        local_outer_sums[1] += n_prev * n_prev;

        states[idx] = std::move(scalar_flux);
      }

      // Sum the errors across all MPI ranks
      double global_outer_sums[2] = {local_outer_sums[0], local_outer_sums[1]};
      if (comm_.size() > 1) {
        ereq = comm_.iallreduce(
          local_outer_sums, global_outer_sums, 2, parallel::MPIOp::SUM);
        ereq.wait();
      }
      error = (global_outer_sums[1] > 0.0)
                ? std::sqrt(global_outer_sums[0] / global_outer_sums[1])
                : std::numeric_limits<double>::max();
      outer_flux_error_history.push_back(error);

      // Update outer-iteration diagnostics exposed to callback_, and invoke
      // it (if set) BEFORE update_convergence_criteria() below -- see
      // solve_eigenvalue()'s identical callback_ placement for the
      // rationale.
      last_num_outer_iterations_ = num_outer_iterations;
      last_outer_error_ = error;
      if (callback_ && i % callback_frequency_ == 0) {
        callback_(*this, *inner_solver);
      }

      // Feed the outer error to the solver's generic convergence hook -- see
      // solve_eigenvalue()'s identical call for the rationale.
      inner_solver->update_convergence_criteria(error);

      if (verbose) {
        for (const auto& sys : local_systems) {
          const auto& x = sys->get_state();
          std::cout << "GID: " << sys->get_gid()
                    << ", Ranks: " << x.as_tt().get_ranks()
                    << ", Compression: " << x.get_compression() << std::endl;
        }

        std::cout << "-- (" << i << "): Scalar Flux L2 Error = " << std::fixed
                  << std::setprecision(10) << error
                  << ", Elapsed Time = " << std::fixed << std::setprecision(3)
                  << static_cast<double>(
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count()) *
                       1e-3
                  << " s" << std::defaultfloat << std::endl;
      }
      if (error < tol) {
        break;
      }
    }

    // Remove the linear systems from GPU
    inner_solver->finalize();

    if (verbose) {
      std::cout << "-- " << (error < tol ? "Converged!" : "Failed to Converge!")
                << std::endl;
    }

    // Build the TransportSolution holding the raw angular flux per local
    // patch; call TransportSolution::compute_scalar_flux() on the result to
    // get a spatial-only field suitable for plotting/averaging.
    auto solution = Solution::create(
      Communicator::world(), mesh_, angular_qset_, xs_server_, config_);
    solution->set_iteration_counts(
      num_outer_iterations, total_inner_iterations);
    // No eigenvalue in a fixed-source solve -- outer_k/outer_k_error carry
    // quiet_NaN() rather than being left empty, so they stay the same
    // length as outer_flux_error_history for callers that zip them. Size
    // captured before any std::move below -- argument evaluation order is
    // unspecified, so reading .size() in the same call as moving from the
    // same vector would be a hazard.
    const size_t num_outer = outer_flux_error_history.size();
    solution->set_convergence_history(
      std::vector<double>(num_outer, std::numeric_limits<double>::quiet_NaN()),
      std::vector<double>(num_outer, std::numeric_limits<double>::quiet_NaN()),
      std::move(outer_flux_error_history), std::move(inner_outer_iter_history),
      std::move(inner_schwarz_error_history));
    for (const auto& sys : local_systems) {
      solution->add_local_patch(sys->get_gid(), sys->get_state());
    }
    return solution;
  }

  /// @brief Partition the Mesh according to the load heuristics. Partitioning
  /// runs METIS_PartGraphKway on rank root_rank.
  /// @param load_heuristics These heuristics compute weights for each local
  /// MeshBlock to help METIS decide the best repartition.
  /// @param root_rank The rank to run METIS on.
  void distribute(
    std::vector<LoadHeuristicPtr> load_heuristics = {}, int root_rank = 0)
  {
    // MPI world size is one (only one MPI rank) -- mesh_'s gid2rank_ is
    // already trivially populated by Mesh::finalize(), nothing to do.
    if (comm_.size() == 1) {
      return;
    }

    // Finalize the heuristics
    for (auto& heuristic : load_heuristics) {
      heuristic->finalize(mesh_);
    }

    // Determine what MeshBlocks will stay on this MPI rank
    auto [local_gids, gid2rank] = load_balancer_.compute_partition(
      std::move(mesh_->build_connectivity_graph()), comm_, load_heuristics,
      root_rank);

    // Restrict the mesh -- cull_blocks() stores gid2rank as mesh_'s own
    // gid2rank_.
    mesh_->cull_blocks(std::move(local_gids), std::move(gid2rank));
    is_distributed_ = true;
  }

  /// @brief Repartition the Mesh according to the load heuristics.
  /// Repartitioning runs ParMETIS_V3_AdaptiveRepart
  /// (https://karypis.github.io/glaros/files/sw/parmetis/manual.pdf).
  /// @param load_heuristics These heuristics compute weights for each local
  /// MeshBlock to help ParMETIS decide the best repartition.
  void redistribute(std::vector<LoadHeuristicPtr> load_heuristics = {})
  {
    // MPI world size is one (only one MPI rank)
    if (comm_.size() == 1) {
      return;
    }

    // Finalize the heuristics
    for (auto& heuristic : load_heuristics) {
      heuristic->finalize(mesh_);
    }

    // Create routing table for sending MeshBlocks around
    const auto& routing_table = load_balancer_.compute_repartition(
      std::move(mesh_->build_connectivity_graph()), comm_, load_heuristics);
  }

  /// @brief Build local systems from assembled patch data and initialize the
  /// given solver with them.
  /// @param solver The solver to initialize.
  /// @param clear_assemblers Clear each patch's assembler after building the
  /// local systems.
  /// @return The global fission source.
  double init_solver(
    const solvers::Solver::Ptr& solver, bool clear_assemblers = true)
  {
    // Clear assemblers
    if (clear_assemblers) {
      for (auto& [gid, pd] : patch_data_) {
        pd.assembler = nullptr;
      }
    }

    // Build the local systems vector in mesh block order
    std::vector<linalg::LinearSystem::Ptr> local_systems;
    local_systems.reserve(patch_data_.size());
    for (const auto& block : mesh_->get_blocks()) {
      local_systems.push_back(patch_data_.at(block->get_gid()).system);
    }

    // Populate the initial guess and compute the initial fission source
    // (k-eigenvalue)
    double k = 0;
    for (auto& sys : local_systems) {
      const auto& interior_op = sys->get_interior_op();
      const auto& n_modes = interior_op.as_tt().get_n_modes();
      const auto& device = interior_op.get_device();
      const auto& dtype = interior_op.get_dtype();

      // Check if this is a fissile system
      linalg::State psi =
        linalg::State::ones(solver->get_state_format(), n_modes, device, dtype);
      const auto& src = sys->get_source();
      if (src && src->is_eigenvalue()) {
        // Update the fission source
        const auto& fsrc = std::static_pointer_cast<linalg::EigenSource>(src);
        fsrc->update(psi, solver->get_eps(), solver->get_max_rank());

        // Compute this patch's contribution
        k += fsrc->get_total_source();
      }

      // Set the initial guess
      sys->set_state(std::move(psi));
    }

    // Determine the total fission source (k-eigenvalue) across all ranks
    parallel::Request kreq;
    double k_global = k;
    if (comm_.size() > 1) {
      kreq = comm_.iallreduce(&k, &k_global, 1, parallel::MPIOp::SUM);
    }

    // Initialize the solver
    solver->init(std::move(local_systems));

    // Wait for the eigenvalue to be sent across ranks
    kreq.wait();
    return k_global;
  }

  // =================================================================
  // Public getters / setters
  /// @return Get the label of the driver.
  const Label& get_label() const noexcept { return label_; }
  /// @return Get the shared pointer to the mesh.
  const Mesh::Ptr& get_mesh() const noexcept { return mesh_; }
  /// @return Get the shared pointer to the cross section library.
  const xs::Server::Ptr& get_server() const noexcept { return xs_server_; }
  /// @return GID -> owning rank, populated by distribute() (or trivially, by
  /// gid -> this rank, if distribute() was never called). Delegates to
  /// mesh_'s own gid2rank_ -- see Mesh::get_gid2rank().
  const std::unordered_map<int64_t, int>& get_gid2rank() const noexcept
  {
    return mesh_->get_gid2rank();
  }
  /// @brief Set a callback invoked once per outer iteration inside
  /// solve_eigenvalue()/solve_fixed_source(), with this driver and the
  /// inner_solver passed to that call, both by const reference -- the
  /// callback body can pull whatever it needs off either (e.g.
  /// last_k()/last_num_outer_iterations() from this driver,
  /// get_local_systems()/get_eps() from the solver). Pass nullptr (the
  /// default) to disable. `frequency` is checked before callback_ is
  /// invoked, so a skipped outer iteration (e.g. a Python callback wrapped
  /// in a GIL acquire by the binding) never pays for the call at all.
  /// @param frequency Invoke callback every `frequency`-th outer iteration
  /// (1 = every iteration, the default). Must be >= 1.
  void set_callback(Callback callback, int frequency = 1)
  {
    if (frequency < 1) {
      throw utils::runtime_error("ttnte::driver::TransportDriver::set_callback",
        "`frequency` must be greater than or equal to 1");
    }
    callback_ = std::move(callback);
    callback_frequency_ = frequency;
  }
  /// @return The eigenvalue as of the most recent outer iteration of the
  /// most recent solve_eigenvalue() call. nullopt for solve_fixed_source()
  /// (no eigenvalue) or before the first outer iteration has run.
  std::optional<double> last_k() const noexcept { return last_k_; }
  /// @return Number of outer iterations the most recent solve_eigenvalue()/
  /// solve_fixed_source() call has run so far.
  int last_num_outer_iterations() const noexcept
  {
    return last_num_outer_iterations_;
  }
  /// @return The scalar-flux-shape relative L2 error as of the most recent
  /// outer iteration of the most recent solve_eigenvalue()/
  /// solve_fixed_source() call.
  double last_outer_error() const noexcept { return last_outer_error_; }

  /// @brief Get the assembler for a specific mesh block GID.
  /// @param gid Global ID of the mesh block.
  /// @throws runtime_error if the assembler was cleared or GID is unknown.
  const Assembler::Ptr& get_assembler(int64_t gid) const
  {
    auto it = patch_data_.find(gid);
    if (it == patch_data_.end()) {
      throw utils::runtime_error(
        "ttnte::driver::TransportDriver::get_assembler",
        "No patch data found for GID " + std::to_string(gid) +
          ". Has assemble() been called?");
    }
    if (!it->second.assembler) {
      throw utils::runtime_error(
        "ttnte::driver::TransportDriver::get_assembler",
        "Assembler for GID " + std::to_string(gid) +
          " was cleared during solve. Set clear_assemblers=false "
          "in DDSolverConfig to retain it.");
    }
    return it->second.assembler;
  }

  /// @brief Get the assembled linear system for a specific mesh block GID.
  /// @param gid Global ID of the mesh block.
  /// @throws runtime_error if GID is unknown or assemble() has not been called.
  const linalg::LinearSystem::Ptr& get_system(int64_t gid) const
  {
    auto it = patch_data_.find(gid);
    if (it == patch_data_.end()) {
      throw utils::runtime_error("ttnte::driver::TransportDriver::get_system",
        "No patch data found for GID " + std::to_string(gid) +
          ". Has assemble() been called?");
    }
    return it->second.system;
  }

  /// @param label The new label of the driver.
  void set_label(const std::string& label)
  {
    label_ = Label::from_string(label);
  }

  /// @brief GID -> this rank's own local patch assembler, NumDim-erased
  /// (base-class handle), for every patch whose assembler hasn't been
  /// cleared. Pass to TransportSolution::compute_patch_balances()/
  /// patch_balance_table()/global_balance() so those methods don't need the
  /// full TransportDriver -- only each local patch's assembler.
  /// @throws Nothing -- patches with a cleared assembler are simply omitted;
  /// callers see TransportSolution's own clear error message if one of
  /// their local GIDs is missing.
  std::unordered_map<int64_t, typename Solution::AssemblerPtr> get_assemblers()
    const
  {
    std::unordered_map<int64_t, typename Solution::AssemblerPtr> result;
    for (const auto& [gid, data] : patch_data_) {
      if (data.assembler) {
        result.emplace(gid, data.assembler);
      }
    }
    return result;
  }
};

} // namespace ttnte::driver
