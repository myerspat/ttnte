#pragma once

#include "ttnte/mesh/mesh.hpp"
#include <memory>
#include <sstream>
#include <torch/extension.h>

namespace py = pybind11;

template<typename BlockType>
void register_Mesh(py::module_& m, const std::string& typestr)
{
  using namespace ttnte::mesh;
  using Mesh = Mesh<BlockType>;
  using MeshPtr = std::shared_ptr<Mesh>;

  py::class_<Mesh, MeshPtr>(m, (typestr + "Mesh").c_str())
    // =================================================================
    // Public constructors
    .def(py::init([](const ttnte::parallel::ParallelContext& mpi_context,
                    std::optional<std::string> label = std::nullopt) {
      return Mesh::create(mpi_context, label);
    }),
      py::arg("mpi_context"), py::arg("label") = py::none())
    .def(py::init(
           [](int my_rank, std::optional<std::string> label = std::nullopt) {
             return Mesh::create(my_rank, label);
           }),
      py::arg("my_rank"), py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def("is_finalized", &Mesh::is_finalized)
    .def("is_connected", &Mesh::is_connected)
    .def("finalize", &Mesh::finalize)
    .def("reserve", &Mesh::reserve, py::arg("size"))
    .def("add_block", &Mesh::add_block, py::arg("new_block"))
    .def("connect", &Mesh::connect, py::arg("tol") = 1e-8)
    .def_static("get_boundary_mapping", &Mesh::get_boundary_mapping,
      py::arg("face_a"), py::arg("face_b"), py::arg("dim_a"), py::arg("dim_b"),
      py::arg("tol") = 1e-8)
    .def("set_axis_aligned_conditions", &Mesh::set_axis_aligned_conditions,
      py::arg("bcplanes"), py::arg("type"), py::arg("tol") = 1e-8)
    .def("build_connectivity_graph", &Mesh::build_connectivity_graph)

    // =================================================================
    // Public overloads
    .def("__repr__",
      [](const Mesh& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      })

    // =================================================================
    // Public Getters / Setters
    .def_property("label", &Mesh::get_label, &Mesh::set_label)
    .def_property_readonly("num_blocks", &Mesh::get_num_blocks)
    .def_property_readonly("blocks", &Mesh::get_blocks)
    .def_property_readonly("bbox", &Mesh::get_bbox);
}
