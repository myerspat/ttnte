#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_BoundaryType(py::module_& m);
void register_assembly_configs(py::module_& m);
void register_dg_first_order_transport_backends(py::module_& m);
void register_dg_assemblers(py::module_& m);
void register_dg_first_order_transport_assemblers(py::module_& m);

void init_physics(py::module_& m)
{
  m.doc() = "Physics module";

  // Register classes
  register_BoundaryType(m);
  register_assembly_configs(m);
  register_dg_first_order_transport_backends(m);
  register_dg_assemblers(m);
  register_dg_first_order_transport_assemblers(m);
}
