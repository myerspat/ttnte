// bindings/cad/bindings.cpp

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// Forward declarations
void register_BSplineBasis(py::module_& m);
void register_Patch(py::module_& m);

PYBIND11_MODULE(cad, m)
{
  m.doc() = "CAD module for B-Spline objects";

  // Register classes in order (basis functions first, since rest depends on it)
  register_BSplineBasis(m);
  register_Patch(m);
}
