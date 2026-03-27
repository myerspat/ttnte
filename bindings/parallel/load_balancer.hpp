#pragma once

#include "../utils/label.hpp"
#include "ttnte/parallel/load_balancer.hpp"
#include <pybind11/pytypes.h>
#include <torch/extension.h>

namespace py = pybind11;

template<typename T>
void register_LoadBalancer(py::module_& m, const std::string& typestr)
{
  using LoadBalancer = ttnte::parallel::LoadBalancer<T>;
  using LoadHeuristicPtr = LoadBalancer::LoadHeuristicPtr;
  register_Label<LoadBalancer>(m, typestr + "LoadBalancer");

  py::class_<LoadBalancer>(m, (typestr + "LoadBalancer").c_str())
    // =================================================================
    // Public constructors
    .def(py::init<int, std::optional<std::string>>(), py::arg("world_size"),
      py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def("compute_repartition", &LoadBalancer::compute_repartition,
      py::arg("conn_graph"), py::arg("comm"),
      py::arg("load_heuristics") = std::vector<LoadHeuristicPtr> {})
    .def("compute_partition", &LoadBalancer::compute_partition,
      py::arg("conn_graph"), py::arg("comm"),
      py::arg("load_heuristics") = std::vector<LoadHeuristicPtr> {},
      py::arg("root_rank") = 0)

    // =================================================================
    // Public getters / setters
    .def_property("label", &LoadBalancer::get_label, &LoadBalancer::set_label)
    .def_property_readonly("block_counts", &LoadBalancer::get_block_counts);
}
