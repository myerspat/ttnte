#include "ttnte/parallel/parallel_context.hpp"
#include "torch/cuda.h"
#include <mpi.h>

namespace ttnte::parallel {

ParallelContext& ParallelContext::instance()
{
  static ParallelContext s_instance;
  return s_instance;
}

void ParallelContext::init()
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

void ParallelContext::finalize()
{
  if (managed_mpi_) {
    MPI_Finalize();
  }
}

void ParallelContext::barrier() const
{
  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace ttnte::parallel
