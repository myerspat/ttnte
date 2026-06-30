#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_BSplineBasis(py::module_& m);
void register_Patch(py::module_& m);

// Initialize CAD module
void init_cad(py::module_& m)
{
  m.doc() = "CAD module for B-Spline objects";

  // Register classes in order (basis functions first, since rest depends on it)
  register_BSplineBasis(m);
  register_Patch(m);
}
