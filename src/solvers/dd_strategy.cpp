#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/utils/exception.hpp"
#include <torch/cuda.h>

namespace ttnte::solvers {

// =================================================================
// Protected constructors
DDStrategy::DDStrategy(bool use_gpu, MemoryPolicy memory_policy)
  : use_gpu_(use_gpu), memory_policy_(memory_policy)
{
  if (use_gpu && !torch::cuda::is_available()) {
    throw utils::runtime_error("ttnte::solvers::DDStrategy::DDStrategy",
      "ttnte was not compiled with the GPU-capable PyTorch version. Ensure "
      "ttnte was compiled with `-DUSE_CUDA=1`.");
  }
}

// =================================================================
// Public methods
void DDStrategy::build_iteration_dag(
  task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const
{
  if (use_gpu_) {
    build_gpu_iteration_dag(dag, local_systems);
  } else {
    build_cpu_iteration_dag(dag, local_systems);
  }
}

void DDStrategy::build_cpu_iteration_dag(
  task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const
{
  throw utils::runtime_error(
    "ttnte::solvers::DDStrategy::build_cpu_iteration_dag",
    "This method is not implemented for this strategy");
}

void DDStrategy::build_gpu_iteration_dag(
  task::TaskGraph& dag, const std::vector<SystemPtr>& local_systems) const
{
  throw utils::runtime_error(
    "ttnte::solvers::DDStrategy::build_gpu_iteration_dag",
    "This method is not implemented for this strategy");
}

} // namespace ttnte::solvers
