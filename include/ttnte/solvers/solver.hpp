#pragma once

#include "ttnte/linalg/format_type.hpp"
#include "ttnte/linalg/linear_system.hpp"
#include <memory>
#include <vector>
namespace ttnte::solvers {

/// @brief Common interface for anything that can solve a set of local linear
/// systems -- a DDSolver (domain decomposition) or a bare LocalSolver (e.g.
/// AMEnSolver, for a single-patch problem), interchangeably from the
/// caller's perspective.
class Solver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<Solver>;
  using Systems = std::vector<linalg::LinearSystem::Ptr>;

  virtual ~Solver() = default;

  // =================================================================
  // Public methods
  /// @brief Register the systems this solver is responsible for.
  virtual void init(const Systems& local_systems) = 0;
  /// @brief Block until one-time worker-thread setup has completed. No-op by
  /// default.
  virtual void wait_for_thread_init() {}
  /// @brief Advance the solve by one call (semantics are solver-specific).
  virtual void step() = 0;
  /// @brief Release any solver-held resources (e.g. GPU memory). No-op by
  /// default.
  virtual void finalize() {}
  /// @brief Generic hook for an external convergence signal. No-op by
  /// default -- solvers with their own internal convergence signal (e.g.
  /// DDSolver) should leave this unoverridden; solvers without one (e.g.
  /// LocalSolver) use it to drive their own forcing.
  /// @param error The latest global convergence error (solver-specific
  /// meaning, e.g. the DD Schwarz current-weighted L2 error).
  /// @param rank_metric An aggregate measure of solution size (e.g. total TT
  /// state element count, summed across every local patch and MPI-reduced),
  /// as of the solve that produced `error`. Defaulted to 0.0 for callers that
  /// don't have or need one; solvers without a rank-aware EnrichmentPolicy
  /// ignore it.
  virtual void update_convergence_criteria(
    double error, double rank_metric = 0.0)
  {}
  /// @brief Seed the solver's tolerance forcing, before the first step(),
  /// with an a-priori estimate of how accurate the starting guess already is.
  /// Forcing schedules start from "error = 1" (loose truncation, tightening
  /// as sweeps proceed), which is right for a cold start but truncates a good
  /// warm start away in its first sweeps. Defaults to
  /// update_convergence_criteria(error), which is the right seed for a bare
  /// LocalSolver; DDSolver overrides it because its update_convergence_
  /// criteria() is deliberately a no-op (its forcing is driven by its own
  /// internal Schwarz error, which the outer flux error must not feed).
  /// @param error Estimated relative error of the starting guess; >= 1.0
  /// leaves the cold-start schedule unchanged.
  virtual void seed_convergence_criteria(double error)
  {
    update_convergence_criteria(error);
  }
  /// @return Number of inner iterations the most recent step() call ran
  /// (e.g. DDSolver's Schwarz sweeps). Defaults to 1 -- a bare LocalSolver's
  /// step() is a single direct solve with no inner iteration loop of its
  /// own.
  virtual int last_num_iterations() const { return 1; }
  /// @return The convergence error at each inner iteration of the most
  /// recent step() call (e.g. DDSolver's per-Schwarz-sweep current-weighted
  /// L2 error), in iteration order. Defaults to empty -- a bare LocalSolver
  /// has no inner iteration loop of its own to report a trajectory for.
  virtual std::vector<double> last_errors() const { return {}; }

  // =================================================================
  // Public getters / setters
  virtual const Systems& get_local_systems() const = 0;
  virtual void set_local_systems(const Systems& local_systems) = 0;
  /// @return The tensor format this solver represents state in.
  virtual linalg::FormatType get_state_format() = 0;
  /// @return The current truncation tolerance of the solver.
  virtual double get_eps() const { return 0.0; }
  /// @return The maximum rank.
  virtual int64_t get_max_rank() const
  {
    return std::numeric_limits<int64_t>::max();
  }
};

} // namespace ttnte::solvers
