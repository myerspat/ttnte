#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/parallel/boundary_communicator.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/solvers/solver_configs.hpp"
#include "ttnte/task/task_graph.hpp"
#include <torch/extension.h>

namespace py = pybind11;

// Trampoline class to allow subclassing DDStrategy from Python.
class PyDDStrategy : public ttnte::solvers::DDStrategy {
public:
  PyDDStrategy(ttnte::solvers::DDSolverConfig config = {})
    : DDStrategy(std::move(config))
  {}

  void build_cpu_iteration_dag(ttnte::task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems,
    const std::unordered_map<int64_t, size_t>& gid_to_local,
    const ttnte::parallel::BoundaryCommunicator& boundary_comms) const override
  {
    PYBIND11_OVERRIDE(void, DDStrategy, build_cpu_iteration_dag, dag,
      local_systems, gid_to_local, boundary_comms);
  }

  void build_gpu_iteration_dag(ttnte::task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems,
    const std::unordered_map<int64_t, size_t>& gid_to_local,
    const ttnte::parallel::BoundaryCommunicator& boundary_comms,
    const ttnte::parallel::StreamPool::Ptr& stream_pool) const override
  {
    PYBIND11_OVERRIDE(void, DDStrategy, build_gpu_iteration_dag, dag,
      local_systems, gid_to_local, boundary_comms, stream_pool);
  }
};

void register_DDStrategy(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<DDStrategy, PyDDStrategy, std::shared_ptr<DDStrategy>>(
    m, "DDStrategy")
    .def(py::init<DDSolverConfig>(), py::arg("config") = DDSolverConfig {})

    // =================================================================
    // Public methods
    .def("build_cpu_iteration_dag", &DDStrategy::build_cpu_iteration_dag,
      py::arg("dag"), py::arg("local_systems"), py::arg("gid_to_local"),
      py::arg("boundary_comms"))
    .def("build_gpu_iteration_dag", &DDStrategy::build_gpu_iteration_dag,
      py::arg("dag"), py::arg("local_systems"), py::arg("gid_to_local"),
      py::arg("boundary_comms"), py::arg("stream_pool"))

    // =================================================================
    // Public getters / setters
    .def_property("config", &DDStrategy::get_config, &DDStrategy::set_config)
    .def("set_local_solver", &DDStrategy::set_local_solver,
      py::arg("local_solver"))

    // Expose the defaults as static constants for Python visibility
    .def_readonly_static("DEFAULT_USE_GPU", &DEFAULT_USE_GPU)
    .def_readonly_static("DEFAULT_MEMORY_POLICY", &DEFAULT_MEMORY_POLICY);
}
