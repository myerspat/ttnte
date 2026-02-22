// bindings/cad/bindings.cpp

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// Forward declarations
void BasisFunctions(py::module_& m);
void register_Bspline(py::module_& m);

PYBIND11_MODULE(cad, m)
{
  m.doc() = "CAD module for B-Spline objects";

  // Register classes in order (basis functions first, since rest depends on it)

  register_BasisFunctions(m);
  register_Bspline(m);
}
