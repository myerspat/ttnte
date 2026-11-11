#include "ttnte/sweep/sweep_config.hpp"
#include "../utils/kwargs_init.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_SweepConfig(py::module_& m)
{
  using namespace ttnte::sweep;

  py::enum_<LocalSolverType>(m, "LocalSolverType")
    .value("BANDED_LU", LocalSolverType::BANDED_LU,
      "Banded LU with partial pivoting (patch-discontinuous patches)")
    .value("DENSE_LU", LocalSolverType::DENSE_LU,
      "Dense LU (single-element, fully discontinuous patches)")
    .value("SYLVESTER", LocalSolverType::SYLVESTER,
      "Kronecker-sum solver on axis-aligned affine patches (one Schur "
      "factorization per patch for all directions and groups); the "
      "partially separable ExtrudedSolver on extruded 3-D patches; banded LU "
      "elsewhere (the default)")
    .value("ITERATIVE", LocalSolverType::ITERATIVE,
      "SYLVESTER on affine patches; preconditioned matrix-free GMRES (sum "
      "factorization) on curved ones")
    .export_values();

  py::enum_<TaskGranularity>(m, "TaskGranularity")
    .value("UNIT", TaskGranularity::UNIT,
      "One task per unit (most concurrency; a few large PDG patches)")
    .value("SUBDOMAIN", TaskGranularity::SUBDOMAIN,
      "One task per angle set sweeping every local unit (least overhead; "
      "many small FDG elements)")
    .export_values();

  py::enum_<GroupIteration>(m, "GroupIteration")
    .value("JACOBI", GroupIteration::JACOBI,
      "Group sets see the previous pass's flux of the other sets")
    .value("GAUSS_SEIDEL", GroupIteration::GAUSS_SEIDEL,
      "Group sets solved in energy order with the latest flux")
    .export_values();

  py::enum_<InnerSolver>(m, "InnerSolver")
    .value("RICHARDSON", InnerSolver::RICHARDSON,
      "Fixed-point (source) iteration within a group set")
    .value("GMRES", InnerSolver::GMRES,
      "Restarted GMRES within a group set (one sweep per product)")
    .export_values();

  py::enum_<TransportSolve>(m, "TransportSolve")
    .value("EXACT", TransportSolve::EXACT, "Sweep every (direction, group)")
    .value("CROSS", TransportSolve::CROSS,
      "Sweep the pairs chosen by a TT-cross (low-rank angular flux)")
    .export_values();

  py::class_<SweepConfig>(m, "SweepConfig")
    // Keyword arguments only, each an attribute below; defaults from C++
    .def(ttnte::bindings::kwargs_init<SweepConfig>())
    .def_readwrite("local_solver", &SweepConfig::local_solver)
    .def_readwrite("store_angular_flux", &SweepConfig::store_angular_flux)
    .def_readwrite("cache_factorizations", &SweepConfig::cache_factorizations)
    .def_readwrite("task_granularity", &SweepConfig::task_granularity)
    .def_readwrite("num_threads", &SweepConfig::num_threads)
    .def_readwrite("directions_per_task", &SweepConfig::directions_per_task)
    .def_readwrite("group_set_size", &SweepConfig::group_set_size)
    .def_readwrite("group_iteration", &SweepConfig::group_iteration)
    .def_readwrite("inner_solver", &SweepConfig::inner_solver)
    .def_readwrite("inner_tol", &SweepConfig::inner_tol)
    .def_readwrite("inner_forcing", &SweepConfig::inner_forcing)
    .def_readwrite("inner_max_iter", &SweepConfig::inner_max_iter)
    .def_readwrite("gmres_restart", &SweepConfig::gmres_restart)
    .def_readwrite("correction_form", &SweepConfig::correction_form)
    .def_readwrite("incremental_transport", &SweepConfig::incremental_transport)
    .def_readwrite("incremental_switch", &SweepConfig::incremental_switch)
    .def_readwrite("build_solution", &SweepConfig::build_solution)
    .def_readwrite("tol", &SweepConfig::tol)
    .def_readwrite("max_iter", &SweepConfig::max_iter)
    .def_readwrite("rho_safety", &SweepConfig::rho_safety)
    .def_readwrite("face_weight_tol", &SweepConfig::face_weight_tol)
    .def_readwrite("verbose", &SweepConfig::verbose)
    .def_readwrite("progress_seconds", &SweepConfig::progress_seconds)
    .def_readwrite("transport", &SweepConfig::transport)
    .def_readwrite("cross_inner_factor", &SweepConfig::cross_inner_factor,
      "CROSS transport solves to this fraction of the within-set tolerance.")
    .def_readwrite("cross_relax", &SweepConfig::cross_relax,
      "Relax each GMRES product's cross accuracy as the residual drops.")
    .def_readwrite("cross_max_tol", &SweepConfig::cross_max_tol,
      "Loosest relaxed cross tolerance.")
    .def_readwrite("cross_threads", &SweepConfig::cross_threads,
      "Intra-op threads of the cross's linear algebra (capped by "
      "num_threads; <= 0: num_threads).")
    .def_readwrite("global_inner_tolerance",
      &SweepConfig::global_inner_tolerance,
      "Within-set tolerances relative to the whole scalar flux: multiplied by "
      "||phi|| / ||phi_s||, capped at GLOBAL_TOLERANCE_CAP (sweep_config.hpp).")
    .def_readwrite("certify", &SweepConfig::certify,
      "CROSS: certify the converged solve with one exact transport solve of "
      "every group set (see SweepDriver.certification_error).")
    .def_readwrite("local_tol", &SweepConfig::local_tol)
    .def_readwrite("local_restart", &SweepConfig::local_restart)
    .def_readwrite("local_max_iter", &SweepConfig::local_max_iter)
    .def_readwrite("cross", &SweepConfig::cross);
}
