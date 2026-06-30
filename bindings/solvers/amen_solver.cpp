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
    .def(py::init([](int nswp, double eps, int rmax, int max_full, int kickrank,
                    int kick2, int local_iterations, int resets, bool verbose,
                    int prec) {
      return AMEnSolver::create(nswp, eps, rmax, max_full, kickrank, kick2,
        local_iterations, resets, verbose, prec);
    }),
      py::arg("nswp") = 22, py::arg("eps") = 1e-10,
      py::arg("rmax") = std::numeric_limits<int>::max(),
      py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
      py::arg("local_iterations") = 40, py::arg("resets") = 2,
      py::arg("verbose") = false, py::arg("prec") = 0)
    .def("solve", &AMEnSolver::solve, py::arg("local_system"),
      py::call_guard<py::gil_scoped_release>());
}
