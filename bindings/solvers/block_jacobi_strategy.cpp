#include "ttnte/solvers/block_jacobi_strategy.hpp"
#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/solvers/solver_configs.hpp"
#include "ttnte/task/task_graph.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_BlockJacobiStrategy(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<BlockJacobiStrategy, DDStrategy, BlockJacobiStrategy::Ptr>(
    m, "BlockJacobiStrategy")
    // =================================================================
    // Public constructors
    .def(py::init([](const DDSolverConfig& config) {
      return BlockJacobiStrategy::create(config);
    }),
      py::arg("config") = DDSolverConfig {})

    // =================================================================
    // Public methods
    .def(
      "build_cpu_compute_dag",
      [](const BlockJacobiStrategy& self, ttnte::task::TaskGraph& dag,
        const ttnte::linalg::LinearSystem::Ptr& local_system) {
        self.build_cpu_compute_dag(dag, local_system);
      },
      py::arg("dag"), py::arg("local_system"))
    .def(
      "build_gpu_compute_dag",
      [](const BlockJacobiStrategy& self, ttnte::task::TaskGraph& dag,
        const ttnte::linalg::LinearSystem::Ptr& local_system,
        const ttnte::parallel::StreamPool::Ptr& stream_pool) {
        self.build_gpu_compute_dag(dag, local_system, stream_pool);
      },
      py::arg("dag"), py::arg("local_system"), py::arg("stream_pool"));
}
