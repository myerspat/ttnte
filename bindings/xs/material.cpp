#include "ttnte/xs/material.hpp"
#include "../utils/label.hpp"
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_Material(py::module_& m)
{
  using Material = ttnte::xs::Material;
  register_Label<Material>(m, "Material");

  py::class_<Material>(m, "Material")
    // =================================================================
    // Constructors
    .def(py::init<std::optional<Material::Label>>(),
      py::arg("label") = std::nullopt)
    .def(py::init<std::string>(), py::arg("label"))

    // =================================================================
    // Methods
    .def("is_finalized", &Material::is_finalized)
    .def("is_fissile", &Material::is_fissile)
    .def("finalize", &Material::finalize)

    // =================================================================
    // Getters / Setters
    .def_property_readonly("label", &Material::get_label)
    .def_property_readonly("num_groups", &Material::get_num_groups)
    .def_property_readonly("num_moments", &Material::get_num_moments)
    .def_property("chi", &Material::get_chi, &Material::set_chi)
    .def_property("total", &Material::get_total, &Material::set_total)
    .def_property(
      "absorption", &Material::get_absorption, &Material::set_absorption)
    .def_property("kappa_fission", &Material::get_kappa_fission,
      &Material::set_kappa_fission)
    .def_property("fission", &Material::get_fission, &Material::set_fission)
    .def_property(
      "nu_fission", &Material::get_nu_fission, &Material::set_nu_fission)
    .def_property(
      "scatter_gtg", &Material::get_scatter_gtg, &Material::set_scatter_gtg);
}
