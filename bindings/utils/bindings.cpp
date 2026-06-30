#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_ParallelContext(py::module_& m);

// Initialize utils module
void init_utils(py::module_& m)
{
  m.doc() = "ttnte.utils module";

  // Register classes
  register_ParallelContext(m);
}
