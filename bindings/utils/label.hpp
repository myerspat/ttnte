#pragma once

#include "ttnte/utils/label.hpp"
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

// Helper function to bind Label for a specific type
template<typename T>
void register_Label(py::module& m, const std::string& typestr)
{
  using namespace ttnte::utils;
  using Class = Label<T>;
  std::string py_class_name = typestr + "Label";

  py::class_<Class>(m, py_class_name.c_str())
    .def(py::init<>())
    .def(py::init<uint64_t>())
    // Static Factories
    .def_static("create_internal", &Class::create_internal)
    .def_static("from_string", &Class::from_string)
    // Inspectors
    .def("is_valid", &Class::is_valid)
    .def("is_user_defined", &Class::is_user_defined)
    .def("to_int", &Class::to_int)
    .def("to_string", &Class::to_string)
    // Operators
    .def("__eq__", &Class::operator==, py::is_operator())
    .def("__ne__", &Class::operator!=, py::is_operator())
    .def("__lt__", &Class::operator<, py::is_operator())
    // The __str__ and __repr__ hooks in Python
    .def("__repr__", [py_class_name](const Class& self) {
      std::stringstream ss;
      ss << py_class_name << "(" << self << ", is_user_defined="
         << ((self.is_user_defined()) ? "True)" : "False)");
      return ss.str();
    });
}
