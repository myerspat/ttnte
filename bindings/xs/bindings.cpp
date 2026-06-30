#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_Material(py::module_& m);
void register_Server(py::module_& m);

// Initialize cross section module
void init_xs(py::module_& m)
{
  m.doc() = "Cross section module";

  // Register classes
  register_Material(m);
  register_Server(m);
}
