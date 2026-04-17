#include "ttnte/solvers/local_solver.hpp"
#include <torch/extension.h>

namespace py = pybind11;

// Trampoline class for virtual method redirection
class PyLocalSolver : public ttnte::solvers::LocalSolver {
public:
  using LocalSolver::LocalSolver; // Inherit constructors

  // Redirect C++ virtual calls to Python
  void solve(const ttnte::linalg::LinearSystem::Ptr& local_system) override
  {
    PYBIND11_OVERRIDE(void, /* Return type */
      LocalSolver,          /* Parent class */
      solve,                /* Name of function */
      local_system          /* Argument(s) */
    );
  }
};

void register_LocalSolver(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<LocalSolver, PyLocalSolver, std::shared_ptr<LocalSolver>>(
    m, "LocalSolver")
    // =================================================================
    // Public methods
    .def("solve", &LocalSolver::solve, py::arg("local_system"),
      py::call_guard<py::gil_scoped_release>());
}
