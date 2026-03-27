#include "transport_driver.hpp"
#include "ttnte/cad/patch.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize the driver module
void init_driver(py::module_& m)
{
  m.doc() = "Driver module";

  // Register classes
  register_TransportDriver<ttnte::cad::Patch>(m, "IGA");
}
