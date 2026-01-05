#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_Operator(py::module_& m);
void register_TTOperator(py::module_& m);
void register_ContractionStep(py::module_& m);
void register_SparseOperator(py::module_& m);
void register_ScatterOperator(py::module_& m);
void register_FissionOperator(py::module_& m);
void register_LinearOperator(py::module_& m);
void register_gmres(py::module_& m);
void register_eig(py::module_& m);

PYBIND11_MODULE(linalg, m)
{
  m.doc() = "C++ backend for ttnte.linalg module";

  register_Operator(m);
  register_TTOperator(m);
  register_ContractionStep(m);
  register_SparseOperator(m);
  register_ScatterOperator(m);
  register_FissionOperator(m);
  register_LinearOperator(m);
  register_gmres(m);
  register_eig(m);
}
