#pragma once

#include "ttnte/mesh/mesh_block.hpp"
#include "ttnte/mesh/mesh_block_boundary.hpp"
#include "ttnte/xs/material.hpp"
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

    .def_property_readonly(
      "ndim", [](const DerivedType& self) { return self.get_ndim(); })
    .def_property_readonly(
      "device", [](const DerivedType& self) { return self.get_device(); })
    .def_property_readonly(
      "dtype", [](const DerivedType& self) { return self.get_dtype(); });
}
