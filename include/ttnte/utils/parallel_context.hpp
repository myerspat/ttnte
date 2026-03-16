#pragma once

#include <mpi.h>
#include <torch/torch.h>

namespace ttnte::utils {

class ParallelContext {
private:
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

  static ParallelContext& instance()
  {
    static ParallelContext s_instance;
    return s_instance;
  }

  void init()
  {
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
      // We initialize if MPI isn't already running
      MPI_Init(nullptr, nullptr);
      managed_mpi_ = true;
    } else {
      // We attach to existing environment (e.g. mpi4py)
      managed_mpi_ = false;
    }

    // Global IDs
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

    // Local node IDs (hardware grouping)
    MPI_Comm node_comm;
    MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &local_rank_);
    MPI_Comm_free(&node_comm);

    // Bind the respective LibTorch device automatically
    if (torch::cuda::is_available()) {
      int num_devices = torch::cuda::device_count();
      device_ = torch::Device(torch::kCUDA, local_rank_ % num_devices);
    }
  }

  void finalize()
  {
    if (managed_mpi_) {
      MPI_Finalize();
    }
  }

  int rank() const noexcept { return rank_; }
  int world_size() const noexcept { return world_size_; }
  int local_rank() const noexcept { return local_rank_; }
  torch::Device device() const noexcept { return device_; }
};

} // namespace ttnte::utils
