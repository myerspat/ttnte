#include "ttnte/sampling/weighting.hpp"
#include "../utils/kwargs_init.hpp"
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

void register_Weighting(py::module_& m)
{
  using namespace ttnte::sampling;

  py::class_<Weighting>(m, "Weighting",
    "Factors of the L2 weighting of stored fibers, a_d e_g R_P psi_{d,g}: "
    "sqrt(w_d) in angle, 1 / sqrt(du_g) in energy and the patch mass "
    "Cholesky factor R_P in space. All enabled by default.")
    // Keyword arguments only, each an attribute below; defaults from C++
    .def(ttnte::bindings::kwargs_init<Weighting>())
    .def_readwrite("angle", &Weighting::angle, "sqrt(w_d) per direction.")
    .def_readwrite("energy", &Weighting::energy,
      "1 / sqrt(du_g) per group (needs lethargy widths).")
    .def_readwrite("space", &Weighting::space,
      "Patch mass Cholesky factor R_P (needs mass bands).")
    .def("any", &Weighting::any, "Whether any factor is enabled.")
    .def("__repr__", [](const Weighting& w) {
      auto b = [](bool v) { return v ? std::string("True") : "False"; };
      return "Weighting(angle=" + b(w.angle) + ", energy=" + b(w.energy) +
             ", space=" + b(w.space) + ")";
    });
}
