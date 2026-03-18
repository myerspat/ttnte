#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void init_utils(py::module_& m);
void init_xs(py::module_& m);
void init_cad(py::module_& m);

PYBIND11_MODULE(ttnte_python, m)
{
  auto m_utils = m.def_submodule("utils");
  init_utils(m_utils);

  auto m_xs = m.def_submodule("xs");
  init_xs(m_xs);

  auto m_cad = m.def_submodule("cad");
  init_cad(m_cad);
}
