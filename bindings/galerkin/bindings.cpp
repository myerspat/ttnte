#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_GalerkinConfig(py::module_& m);
void register_GalerkinIteration(py::module_& m);
void register_GalerkinStage(py::module_& m);
void register_GalerkinRun(py::module_& m);
void register_GreedyGalerkin(py::module_& m);

// Initialize the galerkin module
void init_galerkin(py::module_& m)
{
  m.doc() = "Residual-greedy Galerkin transport solves (Guo & Peng)";

  // Register classes
  register_GalerkinConfig(m);
  register_GalerkinIteration(m);
  register_GalerkinStage(m);
  register_GalerkinRun(m);
  register_GreedyGalerkin(m);
}
