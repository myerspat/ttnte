#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_BoundaryType(py::module_& m);

void init_physics(py::module_& m)
{
  m.doc() = "Physics module";

  // Register classes
  register_BoundaryType(m);
}
