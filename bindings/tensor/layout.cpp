#include "ttnte/tensor/layout.hpp"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_Layout(py::module_& m)
{
  using namespace ttnte::tensor;

  py::class_<Layout>(m, "Layout",
    "Order of the indices of a tensor train and their grouping into cores. "
    "Built from a string (cores separated by '|', indices of a core by ',', "
    "e.g. 'polar|azimuthal|x,y|energy') or from a list of lists of Index.")
    .def(py::init<const std::string&>(), py::arg("spec"))
    .def(py::init<std::vector<Layout::Core>>(), py::arg("cores"))
    .def_static("canonical", &Layout::canonical, py::arg("num_dim"),
      py::arg("split_space") = true)
    .def("validate", &Layout::validate, py::arg("num_dim"))
    .def("merge_space", &Layout::merge_space)
    .def("with_sizes", &Layout::with_sizes, py::arg("sizes"))
    .def("core_of", &Layout::core_of, py::arg("dim"))
    .def("contains", &Layout::contains, py::arg("dim"))
    .def("core_size", &Layout::core_size, py::arg("k"))
    .def("indices", &Layout::indices)
    .def_property_readonly("cores", &Layout::cores)
    .def_property_readonly("num_cores", &Layout::num_cores)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def("__str__", &Layout::to_string)
    .def("__repr__",
      [](const Layout& l) { return "Layout('" + l.to_string() + "')"; });

  py::implicitly_convertible<std::string, Layout>();
}
