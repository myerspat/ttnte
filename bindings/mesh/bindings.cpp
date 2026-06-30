#include "mesh.hpp"
#include "ttnte/cad/patch.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_ConnectivityGraph(py::module_& m);
void register_Boundary(py::module_& m);

void init_mesh(py::module_& m)
{
  m.doc() = "Mesh module";

  // Register classes
  register_ConnectivityGraph(m);
  register_Boundary(m);
  register_Mesh<ttnte::cad::Patch>(m, "IGA");
}
