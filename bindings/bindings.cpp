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

PYBIND11_MODULE(ttnte_python, m)
{
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
}
