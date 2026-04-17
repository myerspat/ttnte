#include "ttnte/solvers/dd_strategy.hpp"
#include <torch/extension.h>

namespace py = pybind11;

// Trampoline class to allow overriding virtual methods in Python if needed
class PyDDStrategy : public ttnte::solvers::DDStrategy {
public:
  using DDStrategy::DDStrategy;

  void build_cpu_iteration_dag(ttnte::task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems) const override
  {
    PYBIND11_OVERRIDE(
      void, DDStrategy, build_cpu_iteration_dag, dag, local_systems);
  }

  void build_gpu_iteration_dag(ttnte::task::TaskGraph& dag,
    const std::vector<SystemPtr>& local_systems) const override
  {
    PYBIND11_OVERRIDE(
      void, DDStrategy, build_gpu_iteration_dag, dag, local_systems);
  }
};

void register_DDStrategy(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<DDStrategy, PyDDStrategy, std::shared_ptr<DDStrategy>>(
    m, "DDStrategy")
    // =================================================================
    // Public methods
    .def("build_iteration_dag", &DDStrategy::build_iteration_dag,
      py::arg("dag"), py::arg("local_systems"),
      "Main entry point to build the task graph for one iteration.")
    .def("build_cpu_iteration_dag", &DDStrategy::build_cpu_iteration_dag,
      py::arg("dag"), py::arg("local_systems"))
    .def("build_gpu_iteration_dag", &DDStrategy::build_gpu_iteration_dag,
      py::arg("dag"), py::arg("local_systems"))

    // =================================================================
    // Public getters / setters
    .def_property_readonly("use_gpu", &DDStrategy::use_gpu)
    .def_property_readonly("memory_policy", &DDStrategy::get_memory_policy)

    .def("set_local_solver", &DDStrategy::set_local_solver,
      py::arg("local_solver"))

    // Expose the defaults as static constants for Python visibility
    .def_readonly_static("DEFAULT_USE_GPU", &DEFAULT_USE_GPU)
    .def_readonly_static("DEFAULT_MEMORY_POLICY", &DEFAULT_MEMORY_POLICY);
}
