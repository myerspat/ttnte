#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_MemoryPolicy(py::module_& m);

void register_LocalSolver(py::module_& m);
void register_AMEnSolver(py::module_& m);

void register_DDSolverConfig(py::module_& m);
void register_DDStrategy(py::module_& m);
void register_BlockJacobiStrategy(py::module_& m);
void register_DDSolver(py::module_& m);

// Initialize the solvers module
void init_solvers(py::module_& m)
{
  m.doc() = "Solvers module";

  // Register classes
  register_MemoryPolicy(m);

  register_LocalSolver(m);
  register_AMEnSolver(m);

  register_DDSolverConfig(m);
  register_DDStrategy(m);
  register_BlockJacobiStrategy(m);
  register_DDSolver(m);
}
