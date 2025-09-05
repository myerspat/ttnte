#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_TTOperator(py::module_& m);
void register_ContractionStep(py::module_& m);
void register_CSROperator(py::module_& m);

PYBIND11_MODULE(linalg, m)
{
  m.doc() = "C++ backend for ttnte.linalg module";

  register_TTOperator(m);
  register_ContractionStep(m);
  register_CSROperator(m);
}
