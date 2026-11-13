#include "ttnte/galerkin/galerkin_config.hpp"
#include "../utils/kwargs_init.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_GalerkinConfig(py::module_& m)
{
  using namespace ttnte::galerkin;

  py::enum_<GalerkinStart>(m, "GalerkinStart")
    .value("RANDOM", GalerkinStart::RANDOM, "per_iter random directions")
    .value("INITIAL_DIRECTIONS", GalerkinStart::INITIAL_DIRECTIONS,
      "The cross's initial directions");

  py::enum_<GalerkinPhiOld>(m, "GalerkinPhiOld")
    .value("AUTO", GalerkinPhiOld::AUTO,
      "PREVIOUS_RUN under RICHARDSON, WITHIN_RUN under GMRES")
    .value("PREVIOUS_RUN", GalerkinPhiOld::PREVIOUS_RUN,
      "The previous run's scalar flux (Guo & Peng under source iteration)")
    .value("WITHIN_RUN", GalerkinPhiOld::WITHIN_RUN,
      "Successive full solves of the same run (Han et al. under GMRES)");

  py::class_<GalerkinConfig>(m, "GalerkinConfig")
    // Keyword arguments only, each an attribute below; defaults from C++
    .def(ttnte::bindings::kwargs_init<GalerkinConfig>())
    .def_readwrite("per_iter", &GalerkinConfig::per_iter,
      "Directions swept per greedy iteration (p).")
    .def_readwrite("candidates", &GalerkinConfig::candidates,
      "Random unswept candidates scored per iteration (q).")
    .def_readwrite("start", &GalerkinConfig::start)
    .def_readwrite("tol", &GalerkinConfig::tol,
      "Relative accuracy: max_j ||r_j|| <= tol RMS_j ||b_j|| over the "
      "candidates, and ||phi - phi_old|| <= tol ||phi||.")
    .def_readwrite("phi_old", &GalerkinConfig::phi_old,
      "Reference of the scalar-flux change test (a GalerkinPhiOld).")
    .def_readwrite("weighting", &GalerkinConfig::weighting,
      "Factors of the norms (a Weighting; all enabled by default). Set "
      "angle=False for runs consistent with Guo & Peng, who weight every "
      "direction equally.")
    .def_readwrite("mgs_eps", &GalerkinConfig::mgs_eps)
    .def_readwrite("mgs_max_projections", &GalerkinConfig::mgs_max_projections)
    .def_readwrite("basis_rtol", &GalerkinConfig::basis_rtol,
      "Snapshots in the span to this relative accuracy append no vector.")
    .def_readwrite("max_iter", &GalerkinConfig::max_iter,
      "Greedy iterations per run (0: until every direction is swept).")
    .def_readwrite("seed", &GalerkinConfig::seed)
    .def_readwrite("keep_angular_flux", &GalerkinConfig::keep_angular_flux);
}
