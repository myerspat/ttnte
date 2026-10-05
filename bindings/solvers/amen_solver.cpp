#include "ttnte/solvers/amen_solver.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_AMEnSolver(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<AMEnSolver, LocalSolver, std::shared_ptr<AMEnSolver>>(
    m, "AMEnSolver")
    // =================================================================
    // Public constructors
    .def(py::init(
           [](int nswp, double eps, double eps_forcing, int max_rank,
             int max_full, int kickrank, int kick2, int local_iterations,
             int resets, bool verbose, ttnte::linalg::AMEnPreconditioner prec,
             ttnte::linalg::AMEnBackend backend,
             ttnte::linalg::AMEnNativeOptions native_opts,
             EnrichmentPolicy::Ptr enrichment_policy, bool preserve_moments,
             double moment_remainder_relaxation, double moment_eps,
             int64_t moment_max_rank, double eps_max) {
             return AMEnSolver::create(nswp, eps, eps_forcing, max_rank,
               max_full, kickrank, kick2, local_iterations, resets, verbose,
               prec, backend, native_opts, std::move(enrichment_policy),
               preserve_moments, moment_remainder_relaxation, moment_eps,
               moment_max_rank, eps_max);
           }),
      py::arg("nswp") = 22, py::arg("eps") = 1e-10,
      py::arg("eps_forcing") = 0.01,
      py::arg("max_rank") = std::numeric_limits<int>::max(),
      py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
      py::arg("local_iterations") = 40, py::arg("resets") = 2,
      py::arg("verbose") = false,
      py::arg("prec") = ttnte::linalg::AMEnPreconditioner::NONE,
      py::arg("backend") = ttnte::linalg::AMEnBackend::NATIVE,
      py::arg("native_opts") = ttnte::linalg::AMEnNativeOptions {},
      py::arg("enrichment_policy") = EnrichmentPolicy::Ptr {},
      py::arg("preserve_moments") = false,
      py::arg("moment_remainder_relaxation") = 1.0,
      py::arg("moment_eps") = -1.0, py::arg("moment_max_rank") = -1,
      py::arg("eps_max") = std::numeric_limits<double>::infinity())
    .def("solve", &AMEnSolver::solve, py::arg("local_system"),
      py::call_guard<py::gil_scoped_release>())
    .def("is_rank_frozen", &AMEnSolver::is_rank_frozen)
    .def_property_readonly(
      "enrichment_policy", &AMEnSolver::get_enrichment_policy)
    .def_property_readonly("eps_max", &AMEnSolver::get_eps_max);
}
