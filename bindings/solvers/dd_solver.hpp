// #pragma once
//
// #include "ttnte/parallel/dd_solver.hpp"
// #include <torch/extension.h>
//
// namespace py = pybind11;
//
// template<typename BlockType>
// void register_DDSolver_template(py::module_& m, const std::string& type_name)
// {
//   using Class = DDSolver<BlockType>;
//   std::string py_class_name = "DDSolver_" + type_name;
//
//   py::class_<Class, Class::Ptr>(m, py_class_name.c_str())
//     // Constructor via lambda to handle the private constructor + factory
//     // pattern
//     .def(py::init([](typename Class::MeshPtr mesh, DDStrategy::Ptr strategy,
//                     int num_streams, std::optional<std::string> label) {
//       return Class::create(
//         std::move(mesh), std::move(strategy), num_streams, label);
//     }),
//       py::arg("mesh"), py::arg("strategy"), py::arg("num_streams") = 16,
//       py::arg("label") = py::none())
//
//     // Logic methods
//     .def("set_systems", &Class::set_systems, py::arg("local_systems"),
//       "Sets the local linear systems for this solver rank.")
//
//     // Getters
//     .def_property_readonly("label", &Class::get_label)
//     .def_property_readonly("strategy", &Class::get_strategy)
//     .def_property_readonly("local_systems", &Class::get_local_systems)
//     .def_property_readonly("world_comm", &Class::get_world_comm)
//     .def_property_readonly("boundary_comms", &Class::get_boundary_comms)
//     .def_property_readonly("stream_pool", &Class::get_stream_pool);
// }
