#include "ttnte/sweep/angular_moments.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

// Forward declarations
void register_SweepConfig(py::module_& m);
void register_PatchDiscretization(py::module_& m);
void register_SweepGraph(py::module_& m);
void register_SweepDriver(py::module_& m);

// Initialize the sweep module
void init_sweep(py::module_& m)
{
  m.doc() = "Discrete-ordinates transport sweep solver (PDG/FDG IGA)";

  // Register classes
  register_SweepConfig(m);
  register_PatchDiscretization(m);
  register_SweepGraph(m);
  register_SweepDriver(m);

  // Free functions
  m.def(
    "discrete_ordinates",
    [](const ttnte::math::QuadratureSet::Ptr& qset, int64_t ndim) {
      return ttnte::sweep::discrete_ordinates(*qset, ndim);
    },
    "Streaming directions (N, ndim), unit vectors (N, 3) and weights (N) of "
    "a quadrature set, in TT core order.",
    py::arg("qset"), py::arg("ndim"));
  m.def(
    "discrete_harmonics",
    [](const ttnte::math::QuadratureSet::Ptr& qset, int64_t ndim,
      int64_t num_orders) {
      return ttnte::sweep::discrete_harmonics(*qset, ndim, num_orders);
    },
    "Real spherical harmonics at the ordinates (N, num_moments) with the "
    "TT scattering kernel's normalization, and the Legendre order of each "
    "moment.",
    py::arg("qset"), py::arg("ndim"), py::arg("num_orders"));
}
