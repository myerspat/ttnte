#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_FormatType(py::module_& m);
void register_matrix_ops(py::module_& m);
void register_ops(py::module_& m);
void register_tt_ops(py::module_& m);
void register_State(py::module_& m);
void register_Operator(py::module_& m);
void register_TTEngine(py::module_& m);
void register_LinearSystem(py::module_& m);

// Initialize linear algebra module
void init_linalg(py::module_& m)
{
  m.doc() = "Linear algebra module";

  // Register classes
  register_FormatType(m);
  register_matrix_ops(m);
  register_ops(m);
  register_tt_ops(m);
  register_State(m);
  register_Operator(m);
  register_TTEngine(m);
  register_LinearSystem(m);
}
