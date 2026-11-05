#pragma once

#include <memory>
#include <pybind11/pybind11.h>
#include <string>

namespace ttnte::bindings {

/// @brief pybind11 constructor for a plain config struct: the C++ defaults,
/// then every keyword argument assigned to the attribute of that name (each
/// bound with def_readwrite). The defaults live only in the C++ header, so
/// the binding cannot drift from them.
/// @return The py::init to pass to class_::def().
template<typename Config>
auto kwargs_init()
{
  namespace py = pybind11;
  return py::init([](const py::kwargs& kwargs) {
    auto config = std::make_unique<Config>();
    py::object self =
      py::cast(config.get(), py::return_value_policy::reference);
    for (const auto& [key, value] : kwargs) {
      const std::string name = py::str(key);
      if (!py::hasattr(self, name.c_str()))
        throw py::type_error("Unknown keyword argument '" + name + "'");
      self.attr(name.c_str()) = value;
    }
    return config;
  });
}

} // namespace ttnte::bindings
