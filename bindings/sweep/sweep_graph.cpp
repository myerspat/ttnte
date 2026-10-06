#include "ttnte/sweep/sweep_graph.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_SweepGraph(py::module_& m)
{
  using namespace ttnte::sweep;

  py::enum_<EdgeState>(m, "EdgeState")
    .value("NONE", EdgeState::NONE)
    .value("DEPENDENCY", EdgeState::DEPENDENCY)
    .value("LAGGED", EdgeState::LAGGED)
    .export_values();

  py::class_<SweepInterface>(m, "SweepInterface")
    .def(py::init([](int64_t first, int64_t second, torch::Tensor forward,
                    torch::Tensor backward, double tolerance) {
      return SweepInterface {
        first, second, std::move(forward), std::move(backward), tolerance};
    }),
      py::arg("first"), py::arg("second"), py::arg("forward_weight"),
      py::arg("backward_weight"), py::arg("tolerance") = 0.0)
    .def_readonly("first", &SweepInterface::first)
    .def_readonly("second", &SweepInterface::second)
    .def_readonly("forward_weight", &SweepInterface::forward_weight)
    .def_readonly("backward_weight", &SweepInterface::backward_weight)
    .def_readonly("tolerance", &SweepInterface::tolerance);

  py::class_<AngleSet>(m, "AngleSet")
    .def_readonly("directions", &AngleSet::directions)
    .def_readonly("forward", &AngleSet::forward)
    .def_readonly("backward", &AngleSet::backward);

  py::class_<SweepChunk>(m, "SweepChunk")
    .def_readonly("units", &SweepChunk::units)
    .def_readonly("upstream", &SweepChunk::upstream);

  py::class_<SweepGraph, SweepGraph::Ptr>(m, "SweepGraph")
    .def(py::init([](int64_t num_nodes, std::vector<SweepInterface> interfaces,
                    int64_t num_directions) {
      return SweepGraph::create(
        num_nodes, std::move(interfaces), num_directions);
    }),
      py::arg("num_nodes"), py::arg("interfaces"), py::arg("num_directions"))
    .def(
      "upstream", &SweepGraph::upstream, py::arg("angle_set"), py::arg("node"))
    .def(
      "topological_order", &SweepGraph::topological_order, py::arg("angle_set"))
    .def("schedule", &SweepGraph::schedule, py::arg("angle_set"),
      py::arg("chunk_of"),
      "Group nodes into sweep tasks for an angle set, merging chunk-level "
      "cycles.")
    .def_property_readonly("num_nodes", &SweepGraph::num_nodes)
    .def_property_readonly("num_directions", &SweepGraph::num_directions)
    .def_property_readonly("interfaces", &SweepGraph::interfaces)
    .def_property_readonly("angle_sets", &SweepGraph::angle_sets)
    .def_property_readonly("direction_set", &SweepGraph::direction_set)
    .def_property_readonly("num_weight_lagged", &SweepGraph::num_weight_lagged)
    .def_property_readonly("num_dfs_lagged", &SweepGraph::num_dfs_lagged);
}
