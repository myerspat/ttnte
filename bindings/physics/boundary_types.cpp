#include "ttnte/physics/boundary_types.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_BoundaryType(py::module_& m)
{
  using namespace ttnte::physics;

  py::enum_<BoundaryType>(m, "BoundaryType")
    .value("UNKNOWN", BoundaryType::UNKNOWN, "Initial/Undefined state")
    .value("INTERNAL", BoundaryType::INTERNAL, "Connected to another MeshBlock")
    .value("VACUUM", BoundaryType::VACUUM, "Free stream/Vacuum boundary")
    .value("REFLECTIVE", BoundaryType::REFLECTIVE, "Mirror/Reflective boundary")
    .value("DEGENERATE", BoundaryType::DEGENERATE,
      "Boundary is degenerate and needs no boundary operators")
    .export_values()
    .def("__str__", [](ttnte::physics::BoundaryType t) {
      return ttnte::physics::to_string(t);
    });

  py::class_<BCPlane>(m, "BCPlane")
    .def(py::init<bool, bool, bool, bool, bool, bool>(),
      py::arg("x_min") = false, py::arg("x_max") = false,
      py::arg("y_min") = false, py::arg("y_max") = false,
      py::arg("z_min") = false, py::arg("z_max") = false)
    .def(py::init([](const std::vector<bool>& active_planes) {
      return BCPlane {
        c10::SmallVector<bool, 6>(active_planes.begin(), active_planes.end())};
    }),
      py::arg("active_planes"))

    // Expose set_condition as a property
    .def_property_readonly("active_planes", [](const BCPlane& self) {
      return std::vector<bool>(
        self.get_active_planes().begin(), self.get_active_planes().end());
    });
}
