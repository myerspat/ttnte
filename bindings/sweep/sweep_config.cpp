#include "ttnte/sweep/sweep_config.hpp"
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

  py::class_<SweepConfig>(m, "SweepConfig")
    .def(
      py::init([](LocalSolverType local_solver, bool store_angular_flux,
                 bool cache_factorizations, TaskGranularity task_granularity,
                 int64_t num_threads, int64_t group_set_size,
                 GroupIteration group_iteration, InnerSolver inner_solver,
                 double inner_tol, double inner_forcing, int64_t inner_max_iter,
                 int64_t gmres_restart, double tol, int64_t max_iter,
                 bool rho_safety, double face_weight_tol, bool verbose) {
        SweepConfig c;
        c.local_solver = local_solver;
        c.store_angular_flux = store_angular_flux;
        c.cache_factorizations = cache_factorizations;
        c.task_granularity = task_granularity;
        c.num_threads = num_threads;
        c.group_set_size = group_set_size;
        c.group_iteration = group_iteration;
        c.inner_solver = inner_solver;
        c.inner_tol = inner_tol;
        c.inner_forcing = inner_forcing;
        c.inner_max_iter = inner_max_iter;
        c.gmres_restart = gmres_restart;
        c.tol = tol;
        c.max_iter = max_iter;
        c.rho_safety = rho_safety;
        c.face_weight_tol = face_weight_tol;
        c.verbose = verbose;
        return c;
      }),
      py::arg("local_solver") = LocalSolverType::BANDED_LU,
      py::arg("store_angular_flux") = false,
      py::arg("cache_factorizations") = false,
      py::arg("task_granularity") = TaskGranularity::UNIT,
      py::arg("num_threads") = 1, py::arg("group_set_size") = 0,
      py::arg("group_iteration") = GroupIteration::GAUSS_SEIDEL,
      py::arg("inner_solver") = InnerSolver::GMRES, py::arg("inner_tol") = 0.0,
      py::arg("inner_forcing") = 0.0, py::arg("inner_max_iter") = 1000,
      py::arg("gmres_restart") = 30, py::arg("tol") = 1e-6,
      py::arg("max_iter") = 1000, py::arg("rho_safety") = true,
      py::arg("face_weight_tol") = 1e-12, py::arg("verbose") = true)
    .def_readwrite("local_solver", &SweepConfig::local_solver)
    .def_readwrite("store_angular_flux", &SweepConfig::store_angular_flux)
    .def_readwrite("cache_factorizations", &SweepConfig::cache_factorizations)
    .def_readwrite("task_granularity", &SweepConfig::task_granularity)
    .def_readwrite("num_threads", &SweepConfig::num_threads)
    .def_readwrite("group_set_size", &SweepConfig::group_set_size)
    .def_readwrite("group_iteration", &SweepConfig::group_iteration)
    .def_readwrite("inner_solver", &SweepConfig::inner_solver)
    .def_readwrite("inner_tol", &SweepConfig::inner_tol)
    .def_readwrite("inner_forcing", &SweepConfig::inner_forcing)
    .def_readwrite("inner_max_iter", &SweepConfig::inner_max_iter)
    .def_readwrite("gmres_restart", &SweepConfig::gmres_restart)
    .def_readwrite("tol", &SweepConfig::tol)
    .def_readwrite("max_iter", &SweepConfig::max_iter)
    .def_readwrite("rho_safety", &SweepConfig::rho_safety)
    .def_readwrite("face_weight_tol", &SweepConfig::face_weight_tol)
    .def_readwrite("verbose", &SweepConfig::verbose);
}
