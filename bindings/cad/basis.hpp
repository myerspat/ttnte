#pragma once

#include "../utils/label.hpp"
#include "ttnte/cad/basis.hpp"

namespace py = pybind11;

template<typename T>
void register_Basis(py::module& m, const std::string& typestr)
{
  using namespace ttnte::cad;
  using Basis = Basis<T>;
  register_Label<Basis>(m, typestr);
  std::string py_class_name = typestr + "Base";

  py::class_<Basis>(m, py_class_name.c_str())
    .def_property_readonly("label", &Basis::get_label);
}
