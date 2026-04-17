#pragma once

#include <torch/extension.h>

namespace ttnte::parallel {

class ParallelContext {
private:
  // =================================================================
  // Private data
  int rank_;
  int world_size_;
  int local_rank_;
  torch::Device device_;
  bool managed_mpi_;

  ParallelContext()
    : rank_(0), world_size_(1), local_rank_(0), device_(torch::kCPU),
      managed_mpi_(false)
  {}

public:
  // Delete copy/move to enforce Singleton
  ParallelContext(const ParallelContext&) = delete;
  void operator=(const ParallelContext&) = delete;

  // =================================================================
  // Public methods
  /// @return Get the instance of the context singleton.
  static ParallelContext& instance();
  /// @brief Initialize the MPI environment.
  void init();
  /// @brief Finalize the MPI environment.
  void finalize();
  /// @brief Create an MPI barrier and stop everything until MPI finishes here.
  void barrier() const;
  /// @brief MPI abort to get all ranks to crash.
  void mpi_abort() const;

  // =================================================================
  // Public getters / setters
  /// @return The MPI rank.
  int rank() const noexcept { return rank_; }
  /// @return The number of MPI ranks.
  int world_size() const noexcept { return world_size_; }
  /// @return The MPI ranks local to this machine.
  int local_rank() const noexcept { return local_rank_; }
  /// @return The GPU device dedicated to this MPI rank.
  torch::Device device() const noexcept { return device_; }
};

} // namespace ttnte::parallel
