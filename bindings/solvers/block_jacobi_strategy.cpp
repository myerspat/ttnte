#include "ttnte/solvers/block_jacobi_strategy.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_BlockJacobiStrategy(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<BlockJacobiStrategy, DDStrategy, BlockJacobiStrategy::Ptr>(
    m, "BlockJacobiStrategy")
    // =================================================================
    // Public constructors
    .def(py::init([](bool use_gpu, MemoryPolicy memory_policy) {
      return BlockJacobiStrategy::create(use_gpu, memory_policy);
    }),
      py::arg("use_gpu") = DEFAULT_USE_GPU,
      py::arg("memory_policy") = DEFAULT_MEMORY_POLICY)

    // =================================================================
    // Public methods
    .def("build_cpu_compute_dag", &BlockJacobiStrategy::build_cpu_compute_dag,
      py::arg("dag"), py::arg("local_system"), py::arg("is_async") = true,
      py::return_value_policy::reference_internal)
    .def("build_gpu_compute_dag", &BlockJacobiStrategy::build_gpu_compute_dag,
      py::arg("dag"), py::arg("local_system"), py::arg("stream_pool"),
      py::arg("is_async") = true, py::return_value_policy::reference_internal);
}
