#include "ttnte/tensor/index.hpp"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

void register_Index(py::module_& m)
{
  using namespace ttnte::tensor;

  py::enum_<Dimension>(m, "Dimension")
    .value("POLAR", Dimension::POLAR)
    .value("AZIMUTHAL", Dimension::AZIMUTHAL)
    .value("X", Dimension::X)
    .value("Y", Dimension::Y)
    .value("Z", Dimension::Z)
    .value("ENERGY", Dimension::ENERGY)
    .value("SPACE", Dimension::SPACE)
    .value("OTHER", Dimension::OTHER)
    .export_values();

  py::class_<Index>(m, "Index")
    .def(py::init<int64_t, std::optional<std::string>>(), py::arg("size") = 0,
      py::arg("label") = std::nullopt,
      "An index of a given size (0: symbolic) with a user or automatic "
      "label")
    .def_static("known", &Index::known, py::arg("dim"), py::arg("size") = 0)
    .def_static(
      "from_name", &Index::from_name, py::arg("name"), py::arg("size") = 0)
    .def_static("polar", &Index::polar, py::arg("size") = 0)
    .def_static("azimuthal", &Index::azimuthal, py::arg("size") = 0)
    .def_static("x", &Index::x, py::arg("size") = 0)
    .def_static("y", &Index::y, py::arg("size") = 0)
    .def_static("z", &Index::z, py::arg("size") = 0)
    .def_static("energy", &Index::energy, py::arg("size") = 0)
    .def_static("space", &Index::space, py::arg("size") = 0)
    .def("resized", &Index::resized, py::arg("size"))
    .def("same_label", &Index::same_label, py::arg("other"))
    .def_property_readonly("size", &Index::size)
    .def_property_readonly("name", &Index::name)
    .def_property_readonly("dimension", &Index::dimension)
    .def_property_readonly("is_symbolic", &Index::is_symbolic)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def("__hash__", [](const Index& i) { return std::hash<Index> {}(i); })
    .def("__repr__", [](const Index& i) {
      std::ostringstream os;
      os << i;
      return os.str();
    });
}
