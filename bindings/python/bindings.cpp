#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/python/package_manager.hpp"
#include <torch/extension.h>

void register_python_cleanup(py::module_& m)
{
  // Create a capsule that finalizes MPI and cleans up pybind11 bindings
  auto cleanup_capsule = py::capsule([]() {
    // Remove all Python packages
    ttnte::python::PackageManager::instance().clear();

    // Finalize MPI safely
    ttnte::parallel::ParallelContext::instance().finalize();
  });

  m.add_object("_module_cleanup_capsule", cleanup_capsule);
}
