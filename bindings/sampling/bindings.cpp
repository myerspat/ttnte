#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_Weighting(py::module_& m);

// Initialize the sampling module
void init_sampling(py::module_& m)
{
  m.doc() = "What the sweep-sampled transport solves (cross, Galerkin) share";

  // Register classes
  register_Weighting(m);
}
