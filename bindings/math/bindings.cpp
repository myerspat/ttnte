#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_special(py::module_& m);
void register_quadrature_set(py::module_& m);

void init_math(py::module_& m)
{
  m.doc() = "Math module";

  register_special(m);
  register_quadrature_set(m);
}
