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

  py::enum_<GalerkinSample>(m, "GalerkinSample")
    .value("DIRECTION", GalerkinSample::DIRECTION,
      "A direction over every group of the set (Guo & Peng's multigroup)")
    .value("PAIR", GalerkinSample::PAIR,
      "A (direction, group) pair, spatial basis (our ablation)");

  py::enum_<SvdNorm>(m, "SvdNorm")
    .value("FROBENIUS", SvdNorm::FROBENIUS,
      "||discarded||_F <= svd_tol ||all||_F: the truncated flux's weighted L2 "
      "error (the cross's rule)")
    .value("NUCLEAR", SvdNorm::NUCLEAR,
      "sum of discarded <= svd_tol sum of all singular values (Guo & Peng's "
      "printed rule)");

  py::enum_<GalerkinPhiOld>(m, "GalerkinPhiOld")
    .value("AUTO", GalerkinPhiOld::AUTO,
      "PREVIOUS_RUN under RICHARDSON, WITHIN_RUN under GMRES")
    .value("PREVIOUS_RUN", GalerkinPhiOld::PREVIOUS_RUN,
      "The previous run's scalar flux (Guo & Peng under source iteration)")
    .value("WITHIN_RUN", GalerkinPhiOld::WITHIN_RUN,
      "Successive full solves of the same run (Han et al. under GMRES)");

  py::enum_<DeimSelect>(m, "DeimSelect")
    .value("DEIM", DeimSelect::DEIM,
      "Han et al.'s Alg. B.1 (depends on the order of V's columns)")
    .value("QDEIM", DeimSelect::QDEIM,
      "Column-pivoted QR of V^T (invariant to V's orthogonal basis changes)");

  py::class_<DeimConfig>(m, "DeimConfig",
    "Han et al. (2026)'s DEIM source iteration after the greedy stage "
    "(RICHARDSON inner solver; with sample = PAIR over the (direction, "
    "group) pairs, ours).")
    .def(ttnte::bindings::kwargs_init<DeimConfig>())
    .def_readwrite("enabled", &DeimConfig::enabled)
    .def_readwrite("switch_tol", &DeimConfig::switch_tol,
      "Switch from the greedy stage once a run changes the scalar flux by at "
      "most this (relative).")
    .def_readwrite("inner_tol", &DeimConfig::inner_tol,
      "Inner-iteration change that triggers the angular update (eq. 3.3; "
      "1e-6 as in Han et al.; 0, ours: the run's relaxed tolerance).")
    .def_readwrite("select", &DeimConfig::select,
      "Row selection (DeimSelect; DEIM by default).")
    .def_readwrite("galerkin_flux", &DeimConfig::galerkin_flux,
      "Ours (ablation): an update's run holds the Galerkin flux X L instead "
      "of the interpolant (Han et al.: the interpolant; False).")
    .def_readwrite("update_on_growth", &DeimConfig::update_on_growth,
      "Ours (safeguard): an inner run whose change exceeds the previous "
      "one's updates the angle at once (Han et al.: none; False).");

  py::class_<GalerkinConfig>(m, "GalerkinConfig")
    // Keyword arguments only, each an attribute below; defaults from C++
    .def(ttnte::bindings::kwargs_init<GalerkinConfig>())
    .def_readwrite("per_iter", &GalerkinConfig::per_iter,
      "Directions swept per greedy iteration (p).")
    .def_readwrite("candidates", &GalerkinConfig::candidates,
      "Random unswept candidates scored per iteration (q).")
    .def_readwrite("sample", &GalerkinConfig::sample,
      "Sampling unit (GalerkinSample; DIRECTION by default).")
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
    .def_readwrite("keep_angular_flux", &GalerkinConfig::keep_angular_flux)
    .def_readwrite("carry_basis", &GalerkinConfig::carry_basis,
      "Start each run from the previous run's basis, rotated and cut at the "
      "run's own tolerance (not part of Guo & Peng: a labelled ablation).")
    .def_readwrite("svd_tol", &GalerkinConfig::svd_tol,
      "Truncated SVD after a run (Guo & Peng's eps_SVD; 0: off): the "
      "smallest rank whose discarded singular values are at most svd_tol "
      "relative, in svd_norm. Reported as the record's truncated_rank; with "
      "keep_angular_flux the held flux is truncated (the moments are not).")
    .def_readwrite("svd_norm", &GalerkinConfig::svd_norm,
      "Norm of the svd_tol cut (SvdNorm; FROBENIUS by default).")
    .def_readwrite("deim", &GalerkinConfig::deim,
      "Han et al.'s DEIM stage (a DeimConfig; off by default).");
}
