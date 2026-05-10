#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/python/package_manager.hpp"
#include <torch/extension.h>

void register_python_cleanup(py::module_& m)
{
  m.def("_python_cleanup", []() {
    // Remove all Python packages
    ttnte::python::PackageManager::instance().clear();
    // Finalize MPI.
    ttnte::parallel::ParallelContext::instance().finalize();
  });

  // Automatically register the cleanup to Python's atexit
  py::module_ atexit = py::module_::import("atexit");
  atexit.attr("register")(m.attr("_python_cleanup"));
}
