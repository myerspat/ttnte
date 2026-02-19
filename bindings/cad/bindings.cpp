// bindings/cad/bindings.cpp

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// Forward declarations
void register_BasisFunctions(py::module_& m);

PYBIND11_MODULE(cad, m)
{
  m.doc() = "CAD module with B-Splines, NURBS, and TH-NURBS";

  // Register classes in order (basis_functions first, since rest depends on it)

  register_BasisFunctions(m);
}
