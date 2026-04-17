#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_State(py::module_& m);
void register_Operator(py::module_& m);
void register_TTEngine(py::module_& m);
void register_TTState(py::module_& m);
void register_TTOperator(py::module_& m);
void register_LinearSystem(py::module_& m);
void register_TTLinearSystem(py::module_& m);

// Initialize linear algebra module
void init_linalg(py::module_& m)
{
  m.doc() = "Linear algebra module";

  // Register classes
  register_State(m);
  register_Operator(m);
  register_TTEngine(m);
  register_TTState(m);
  register_TTOperator(m);
  register_LinearSystem(m);
  register_TTLinearSystem(m);
}
