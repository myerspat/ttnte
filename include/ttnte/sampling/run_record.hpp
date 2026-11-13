#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace ttnte::sampling {

/// @brief Record of one sampled transport solve (one run of a
/// SampledTransport): what every sampler reports. Samplers derive their own
/// records to add what is theirs (a cross's half-sweeps, a Galerkin run's
/// greedy iterations), so the solver keeps one list of records whatever the
/// sampler.
struct RunRecord {
  using Ptr = std::shared_ptr<RunRecord>;

  virtual ~RunRecord() = default;

  /// Relative tolerance the run was asked for.
  double tolerance = 0.0;
  /// Fibers it swept.
  int64_t fibers = 0;
  /// Ranks of its approximation.
  std::vector<int64_t> ranks;
  /// Its stopping test's error estimate (judged against the tolerance).
  double error = 0.0;
  /// Whether it met its tolerance.
  bool converged = false;
  /// Wall seconds of the run, and of its oracle requests.
  double seconds = 0.0;
  double sweep_seconds = 0.0;
};

} // namespace ttnte::sampling
