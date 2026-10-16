#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_CrossConfig(py::module_& m);
void register_LayoutCross(py::module_& m);

// Initialize the cross module
void init_cross(py::module_& m)
{
  m.doc() = "Sweep-sampled TT-cross of the angular flux";

  // Register classes
  register_CrossConfig(m);
  register_LayoutCross(m);
}
