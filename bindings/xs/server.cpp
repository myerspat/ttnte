#include "ttnte/xs/server.hpp"
#include "../utils/label.hpp"
#include <memory>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_Server(py::module_& m)
{
  using Server = ttnte::xs::Server;
  using Material = ttnte::xs::Material;
  using ServerPtr = std::shared_ptr<Server>;
  register_Label<Server>(m, "Server");

  py::class_<Server, ServerPtr>(m, "Server")
    // =================================================================
    // Public constructors
    .def(
      py::init<std::optional<Server::Label>>(), py::arg("label") = std::nullopt)
    .def(py::init<std::string>(), py::arg("label"))

    // =================================================================
    // Public methods
    .def("is_finalized", &Server::is_finalized)
    .def("finalize", &Server::finalize)
    .def("add_material", &Server::add_material, py::arg("mat"))

    .def("get_material",
      py::overload_cast<const Material::Label::ID&>(
        &Server::get_material, py::const_),
      py::arg("mat_id"), py::return_value_policy::reference_internal)

    .def("get_material",
      py::overload_cast<const Material::Label&>(
        &Server::get_material, py::const_),
      py::arg("mat_label"), py::return_value_policy::reference_internal)

    // =================================================================
    // Getters
    .def_property_readonly(
      "label", &Server::get_label, py::return_value_policy::reference_internal)
    .def_property_readonly("num_groups", &Server::get_num_groups)
    .def_property_readonly("num_moments", &Server::get_num_moments)
    .def_property_readonly("num_materials", &Server::get_num_materials)
    .def_property_readonly("material_ids",
      [](const Server& s) {
        const auto& c10_vec = s.get_material_ids();
        return std::vector<Material::Label::ID>(c10_vec.begin(), c10_vec.end());
      })
    .def_property_readonly("material_map", &Server::get_material_map,
      py::return_value_policy::reference_internal);
}
