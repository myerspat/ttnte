#include "ttnte/python/package_manager.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
// void init_utils(py::module_& m);
void init_xs(py::module_& m);
void init_cad(py::module_& m);
void init_mesh(py::module_& m);
void init_physics(py::module_& m);
void init_task(py::module_& m);
void init_parallel(py::module_& m);
void init_driver(py::module_& m);
void init_linalg(py::module_& m);
void init_solvers(py::module_& m);
void init_sweep(py::module_& m);
void init_math(py::module_& m);

void register_python_cleanup(py::module_& m);

PYBIND11_MODULE(ttnte_python, m)
{
  // Eagerly initialize so the numpy/torchtt imports happen now, not during
  // interpreter shutdown when sys.meta_path is already None.
  ttnte::python::PackageManager::instance();

  // auto m_utils = m.def_submodule("utils");
  // init_utils(m_utils);

  auto m_parallel = m.def_submodule("parallel");
  init_parallel(m_parallel);

  auto m_xs = m.def_submodule("xs");
  init_xs(m_xs);

  auto m_cad = m.def_submodule("cad");
  init_cad(m_cad);

  auto m_mesh = m.def_submodule("mesh");
  init_mesh(m_mesh);

  auto m_physics = m.def_submodule("physics");
  init_physics(m_physics);

  auto m_linalg = m.def_submodule("linalg");
  init_linalg(m_linalg);

  auto m_task = m.def_submodule("task");
  init_task(m_task);

  auto m_driver = m.def_submodule("driver");
  init_driver(m_driver);

  auto m_solvers = m.def_submodule("solvers");
  init_solvers(m_solvers);

  // AMEnBackend/AMEnEnrichmentMode/AMEnPreconditioner/AMEnNativeOptions are
  // defined once in ttnte::linalg, since they're consumed directly by
  // linalg::amen's own numerics (amen_sweep, gmres_solve,
  // FoldedLocalOperator) -- moving them to ttnte::solvers would make linalg
  // depend on solvers, inverting the existing solvers -> linalg dependency
  // direction. They're also the options callers reach for constantly when
  // configuring an AMEnSolver (ttnte.solvers), so alias the same Python type
  // objects into ttnte.solvers too instead of requiring a separate
  // `import ttnte.linalg` just for these.
  m_solvers.attr("AMEnBackend") = m_linalg.attr("AMEnBackend");
  m_solvers.attr("AMEnEnrichmentMode") = m_linalg.attr("AMEnEnrichmentMode");
  m_solvers.attr("AMEnPreconditioner") = m_linalg.attr("AMEnPreconditioner");
  m_solvers.attr("AMEnNativeOptions") = m_linalg.attr("AMEnNativeOptions");

  auto m_sweep = m.def_submodule("sweep");
  init_sweep(m_sweep);

  auto m_math = m.def_submodule("math");
  init_math(m_math);

  register_python_cleanup(m);
}
