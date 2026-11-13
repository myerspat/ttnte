#include "ttnte/galerkin/greedy_galerkin.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_GreedyGalerkin(py::module_& m)
{
  using namespace ttnte::galerkin;

  py::class_<GreedyGalerkin::IterationRecord>(m, "GalerkinIteration",
    "One greedy iteration: rank and fibers after its sweep, the residual "
    "test's ratio and the scalar-flux change (NaN where not computed).")
    .def_readonly("rank", &GreedyGalerkin::IterationRecord::rank)
    .def_readonly("fibers", &GreedyGalerkin::IterationRecord::fibers)
    .def_readonly("residual", &GreedyGalerkin::IterationRecord::residual)
    .def_readonly("phi_change", &GreedyGalerkin::IterationRecord::phi_change);

  py::class_<GreedyGalerkin::RunRecord, ttnte::sampling::RunRecord,
    std::shared_ptr<GreedyGalerkin::RunRecord>>(m, "GalerkinRun",
    "One Galerkin run (transport solve): the common record (a RunRecord), "
    "its greedy iterations, and its snapshots not appended (basis_rtol, a "
    "full basis) or appended with an orthogonality failure.")
    .def_readonly("history", &GreedyGalerkin::RunRecord::history)
    .def_readonly(
      "dropped_snapshots", &GreedyGalerkin::RunRecord::dropped_snapshots)
    .def_readonly("orthogonality_failures",
      &GreedyGalerkin::RunRecord::orthogonality_failures);

  // Built by the sweep solver (TransportSolve.GALERKIN): it needs the
  // solver's projected transport operator
  py::class_<GreedyGalerkin, GreedyGalerkin::Ptr>(m, "GreedyGalerkin")
    .def("run", &GreedyGalerkin::run)
    .def("set_tolerance", &GreedyGalerkin::set_tolerance, py::arg("tol"))
    .def("moments", &GreedyGalerkin::moments,
      "(num_moments, G, num_space) flux moments of the last run.")
    .def("evaluate", &GreedyGalerkin::evaluate, py::arg("directions"),
      py::arg("groups"), py::arg("space"))
    .def("release", &GreedyGalerkin::release)
    .def_property_readonly("tolerance", &GreedyGalerkin::tolerance)
    .def_property_readonly("holds_flux", &GreedyGalerkin::holds_flux)
    .def_property_readonly("num_fibers", &GreedyGalerkin::num_fibers)
    .def_property_readonly("ranks", &GreedyGalerkin::ranks)
    .def_property_readonly("error", &GreedyGalerkin::error)
    .def_property_readonly("converged", &GreedyGalerkin::converged)
    .def_property_readonly("seconds", &GreedyGalerkin::seconds)
    .def_property_readonly("sweep_seconds", &GreedyGalerkin::sweep_seconds)
    .def_property_readonly("timings", &GreedyGalerkin::timings)
    .def_property_readonly("weighting", &GreedyGalerkin::weighting)
    .def_property_readonly("config", &GreedyGalerkin::config,
      "The options in effect (phi_old resolved).")
    .def_property_readonly("history", &GreedyGalerkin::history)
    .def_property_readonly(
      "orthogonality_failures", &GreedyGalerkin::orthogonality_failures)
    .def_property_readonly(
      "dropped_snapshots", &GreedyGalerkin::dropped_snapshots)
    .def_property_readonly("swept", &GreedyGalerkin::swept)
    .def_property_readonly("record", &GreedyGalerkin::record,
      "The last run's record (a GalerkinRun)")
    .def_property_readonly(
      "sweep_calls",
      [](const GreedyGalerkin& self) {
        py::list out;
        for (const auto& c : self.sweep_calls())
          out.append(py::dict(py::arg("phase") = c.phase,
            py::arg("fibers") = c.fibers, py::arg("blocks") = c.blocks,
            py::arg("directions") = c.directions,
            py::arg("seconds") = c.seconds));
        return out;
      },
      "Every sweep request of every run: phase (init, greedy), new fibers, "
      "sweep blocks, distinct directions, seconds");
}
