#include "heuristics.hpp"
#include "load_balancer.hpp"
#include "ttnte/cad/patch.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_RoutingTable(py::module_& m);
void register_ParallelContext(py::module_& m);
void register_Request(py::module_& m);
void register_Communicator(py::module_& m);

// Initialize parallel module
void init_parallel(py::module_& m)
{
  m.doc() = "Parallel module";

  // Register classes
  register_ParallelContext(m);
  register_RoutingTable(m);
  register_Request(m);
  register_Communicator(m);
  register_LoadBalancer<ttnte::cad::Patch>(m, "IGA");
  register_heuristics<ttnte::cad::Patch>(m, "IGA");
}
