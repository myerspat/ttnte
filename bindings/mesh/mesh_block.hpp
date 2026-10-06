#pragma once

#include "ttnte/mesh/mesh_block.hpp"
#include "ttnte/mesh/mesh_block_boundary.hpp"
#include "ttnte/physics/fixed_source.hpp"
#include "ttnte/xs/material.hpp"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

template<typename DerivedType, typename... Options>
void register_MeshBlock(py::class_<DerivedType, Options...>& py_class)
{
  using namespace ttnte::mesh;
  using Base = ttnte::mesh::MeshBlock<DerivedType>;

  py_class
    // =================================================================
    // Public methods
    .def("is_finalized",
      [](const DerivedType& self) { return self.is_finalized(); })
    .def(
      "finalize", [](DerivedType& self) { self.finalize(); },
      "Check the data and make this block immutable.")

    .def(
      "add_connection",
      [](DerivedType& self, size_t dim, bool is_upper,
        const NeighborInfo& ninfo) {
        self.add_connection(dim, is_upper, ninfo);
      },
      py::arg("dim"), py::arg("is_upper"), py::arg("ninfo"))

    .def(
      "get_boundary",
      [](DerivedType& self, size_t dim, bool is_upper) {
        return self.get_boundary(dim, is_upper);
      },
      py::arg("dim"), py::arg("is_upper"))

    .def(
      "get_bbox",
      [](const DerivedType& self, double epsilon) {
        return self.get_bbox(epsilon);
      },
      py::arg("epsilon") = 0.0)

    .def(
      "get_numel",
      [](const DerivedType& self, size_t dim) { return self.get_numel(dim); },
      py::arg("dim"))
    .def(
      "get_boundary_info",
      [](DerivedType& self, size_t dim, bool is_upper) {
        return self.get_boundary_info(dim, is_upper);
      },
      py::arg("dim"), py::arg("is_upper"))
    .def(
      "set_boundary_type",
      [](DerivedType& self, size_t dim, bool is_upper,
        ttnte::physics::BoundaryType type) {
        self.set_boundary_type(dim, is_upper, type);
      },
      py::arg("dim"), py::arg("is_upper"), py::arg("type"))
    .def(
      "set_boundary_albedo",
      [](DerivedType& self, size_t dim, bool is_upper, double albedo) {
        self.set_boundary_albedo(dim, is_upper, albedo);
      },
      py::arg("dim"), py::arg("is_upper"), py::arg("albedo"),
      "Set the specular albedo in [0, 1] of a boundary face (incoming "
      "angular flux = albedo * mirrored outgoing). Only valid on REFLECTIVE "
      "faces; checked at assembly.")
    .def(
      "get_boundary_albedo",
      [](const DerivedType& self, size_t dim, bool is_upper) {
        return self.get_boundary_albedo(dim, is_upper);
      },
      py::arg("dim"), py::arg("is_upper"))
    .def(
      "set_boundary_source",
      [](DerivedType& self, size_t dim, bool is_upper,
        ttnte::physics::FixedSource source) {
        self.set_boundary_source(dim, is_upper, std::move(source));
      },
      py::arg("dim"), py::arg("is_upper"), py::arg("source"))
    .def("get_boundary_info",
      [](DerivedType& self) {
        const auto& boundaries = self.get_boundary_info();
        return std::vector<BoundaryInfo>(boundaries.begin(), boundaries.end());
      })

    // =================================================================
    // Public Getters
    .def("get_numel", [](const DerivedType& self) { return self.get_numel(); })
    .def_property(
      "label", [](const DerivedType& self) { return self.get_label(); },
      [](
        DerivedType& self, const std::string& label) { self.set_label(label); })
    .def_property(
      "fill",
      [](const DerivedType& self) {
        return self.template get_fill<ttnte::xs::Material>();
      },
      [](DerivedType& self, const ttnte::xs::Material::Label& mat) {
        self.set_fill(mat);
      })
    .def_property(
      "fill_id", [](const DerivedType& self) { return self.get_fill_id(); },
      [](DerivedType& self, const uint64_t& fill_id) {
        self.set_fill_id(fill_id);
      })
    .def_property(
      "source", [](const DerivedType& self) { return self.get_fixed_source(); },
      [](DerivedType& self, ttnte::physics::FixedSource source) {
        self.set_source(std::move(source));
      })

    .def_property_readonly(
      "ndim", [](const DerivedType& self) { return self.get_ndim(); })
    .def_property_readonly(
      "device", [](const DerivedType& self) { return self.get_device(); })
    .def_property_readonly(
      "dtype", [](const DerivedType& self) { return self.get_dtype(); })
    .def_property_readonly(
      "gid", [](const DerivedType& self) { return self.get_gid(); });
}
