// bindings/cad/bindings.cpp

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// Forward declarations
void register_Patch(py::module_& m);

PYBIND11_MODULE(cad, m) {
    m.doc() = "CAD module with B-Splines, NURBS, and TH-NURBS";

    // Register classes in order (NURBS before Patch, since Patch depends on it)

    register_patch(m);
}
