#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_FormatType(py::module_& m);
void register_AMEnConfig(py::module_& m);
void register_amen_internal(py::module_& m);
void register_matrix_ops(py::module_& m);
void register_banded_lu(py::module_& m);
void register_dense_lu(py::module_& m);
void register_ops(py::module_& m);
void register_tt_ops(py::module_& m);
void register_State(py::module_& m);
void register_Operator(py::module_& m);
void register_TTEngine(py::module_& m);
void register_NeighborCoupling(py::module_& m);
void register_Source(py::module_& m);
void register_EigenSource(py::module_& m);
void register_LinearSystem(py::module_& m);

// Initialize linear algebra module
void init_linalg(py::module_& m)
{
  m.doc() = "Linear algebra module";

  // Register classes
  register_FormatType(m);
  register_AMEnConfig(m);
  register_amen_internal(m);
  register_matrix_ops(m);
  register_banded_lu(m);
  register_dense_lu(m);
  register_ops(m);
  register_tt_ops(m);
  register_State(m);
  register_Operator(m);
  register_TTEngine(m);
  register_NeighborCoupling(m);
  register_Source(m);
  register_EigenSource(m);
  register_LinearSystem(m);
}
