#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/utils/exception.hpp"
#include <torch/cuda.h>

namespace ttnte::solvers {

// =================================================================
// Protected constructors
DDStrategy::DDStrategy(DDSolverConfig config)
  : config_(std::move(config)),
    tt_config_(std::make_shared<linalg::TTConfig>(config_.rounding))
{
  if (config_.use_gpu && !torch::cuda::is_available()) {
    throw utils::runtime_error("ttnte::solvers::DDStrategy::DDStrategy",
      "ttnte was not compiled with the GPU-capable PyTorch version. Ensure "
      "ttnte was compiled with `-DUSE_CUDA=1`.");
  }
}

// =================================================================
// Public methods
void DDStrategy::build_cpu_iteration_dag(task::TaskGraph& dag,
  const std::vector<SystemPtr>& local_systems,
  const std::unordered_map<int64_t, size_t>& gid_to_local,
  const parallel::BoundaryCommunicator& boundary_comms) const
{
  throw utils::runtime_error(
    "ttnte::solvers::DDStrategy::build_cpu_iteration_dag",
    "This method is not implemented for this strategy");
}

void DDStrategy::build_gpu_iteration_dag(task::TaskGraph& dag,
  const std::vector<SystemPtr>& local_systems,
  const std::unordered_map<int64_t, size_t>& gid_to_local,
  const parallel::BoundaryCommunicator& boundary_comms,
  const parallel::StreamPool::Ptr& stream_pool) const
{
  throw utils::runtime_error(
    "ttnte::solvers::DDStrategy::build_gpu_iteration_dag",
    "This method is not implemented for this strategy");
}

} // namespace ttnte::solvers
