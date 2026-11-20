#pragma once

#include "ttnte/sampling/fiber_store.hpp"
#include "ttnte/sampling/run_record.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <vector>

namespace ttnte::sampling {

/// @brief A transport solve of one group set approximated from sampled
/// sweeps: the sweeper's stand-in for an exact sweep when the transport
/// solve is sampled (TransportSolve::CROSS: cross::LayoutCross;
/// TransportSolve::GALERKIN: galerkin::GreedyGalerkin).
///
/// A run builds a low-rank angular flux of the set's groups from fibers
/// swept through the problem's oracle (see FiberStore), to a relative
/// tolerance set before each run. The sweeper then reads from it everything
/// an exact sweep leaves behind: the moments, the flux at the faces of the
/// lagged edges and, if stored, the angular flux. Groups are local to the
/// set (0 ... G - 1), space is the fiber index (every patch, concatenated).
/// Samplers differ in which fibers they request, what they fit and how they
/// decide they are done; the records below are common to all of them, and so
/// is the state behind them (the store, the last run's error, convergence and
/// seconds, the phase timings), which this base holds for its subclasses.
class SampledTransport {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<SampledTransport>;

  virtual ~SampledTransport() = default;

  // =================================================================
  // Public methods
  /// @brief Approximate the transport solve for the current state of the
  /// oracle (sources and lagged traces set by the caller).
  virtual void run() = 0;
  /// @brief Set the relative tolerance of the next runs.
  virtual void set_tolerance(double tol) = 0;
  /// @return (num_moments, G, num_space) flux moments of the last run.
  virtual torch::Tensor moments() const = 0;
  /// @return (|directions|, |groups|, |space|) angular flux values of the
  /// last run.
  virtual torch::Tensor evaluate(const std::vector<int64_t>& directions,
    const std::vector<int64_t>& groups,
    const std::vector<int64_t>& space) const = 0;
  /// @brief Drop the low-rank flux of the last run.
  virtual void release() = 0;
  /// @brief Compress the held flux once the caller has read what the
  /// iteration needs from it (moments, lagged traces), before it is stored
  /// or kept as the angular flux. Default: nothing.
  virtual void compress() {}
  /// @return Whether the last run may end the within-set iteration: false
  /// while a sampler iterates on a lagged representation the solver cannot
  /// see (Han et al.'s DEIM inner loop before its angular update), so that
  /// the solver does not stop on it. Default: true.
  virtual bool settled() const { return true; }
  /// @brief Set the relative change at which the solver stops iterating on
  /// this sampler's runs (its within-set tolerance), for settled(). Default:
  /// ignored.
  virtual void set_stop_tolerance(double /*tol*/) {}
  /// @brief Start a new sequence of runs (a new solve): drop everything a
  /// sampler carries from run to run (warm starts, references of its
  /// stopping tests, a stage, its random streams), so that the sequence
  /// runs as on a newly built sampler. Default: nothing carried.
  virtual void restart() {}

  // =================================================================
  // Public getters
  /// @return The relative tolerance of the next run.
  virtual double tolerance() const = 0;
  /// @return Whether the low-rank flux of the last run is held (moments()
  /// and evaluate() need it).
  virtual bool holds_flux() const = 0;
  /// @return Whether the flux is to be kept after the solve that last ran
  /// it (for the solver's angular flux), instead of released.
  virtual bool keeps_flux() const = 0;
  /// @return The ranks of the last run's approximation.
  virtual std::vector<int64_t> ranks() const = 0;
  /// @return The last run's record (a sampler's own RunRecord subclass).
  virtual RunRecord::Ptr record() const = 0;

  /// @return Fibers swept by the last run.
  int64_t num_fibers() const noexcept { return store_->num_swept(); }
  /// @return The relative error estimate of the last run's stopping test
  /// (judged against the tolerance, or a multiple of it; each sampler
  /// documents its own).
  double error() const noexcept { return error_; }
  /// @return Whether the last run met its tolerance.
  bool converged() const noexcept { return converged_; }
  /// @return Wall seconds of the last run, and of its oracle requests.
  double seconds() const noexcept { return seconds_; }
  double sweep_seconds() const noexcept { return store_->sweep_seconds(); }
  /// @return Seconds per phase, accumulated over every run: the sampler's
  /// own phases and the oracle's ("sweep", "weight_cpu", from the store).
  std::map<std::string, double> timings() const
  {
    auto out = timings_;
    for (const auto& [phase, sec] : store_->timings())
      out[phase] += sec;
    return out;
  }
  /// @return Every oracle request of every run.
  const std::vector<FiberStore::SweepCall>& sweep_calls() const noexcept
  {
    return store_->sweep_calls();
  }
  /// @return The weighting factors in effect: those configured that the
  /// problem has the data for.
  const Weighting& weighting() const noexcept { return store_->weighting(); }

protected:
  // =================================================================
  // Protected data
  /// The swept fibers (created by the subclass's constructor).
  FiberStore::Ptr store_;
  /// The last run's error estimate, convergence and wall seconds.
  double error_ = 0.0;
  bool converged_ = false;
  double seconds_ = 0.0;
  /// Seconds per phase of run() outside the oracle, accumulated over every
  /// run (the oracle's are the store's).
  std::map<std::string, double> timings_;

  // =================================================================
  // Protected methods
  /// @brief Fill the fields every sampler reports from the getters above.
  void fill_record(RunRecord& record) const
  {
    record.tolerance = tolerance();
    record.fibers = num_fibers();
    record.ranks = ranks();
    record.error = error();
    record.converged = converged();
    record.seconds = seconds();
    record.sweep_seconds = sweep_seconds();
  }
};

} // namespace ttnte::sampling
