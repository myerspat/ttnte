#pragma once

#include "../utils/label.hpp"
#include "ttnte/driver/transport_driver.hpp"
#include <torch/extension.h>

namespace py = pybind11;

template<typename BlockType>
void register_TransportDriver(py::module_& m, const std::string& typestr)
{
  using TransportDriver = ttnte::driver::TransportDriver<BlockType>;
  using LoadHeuristicPtr =
    ttnte::driver::TransportDriver<BlockType>::LoadHeuristicPtr;
  std::string class_name = typestr + "TransportDriver";

  register_Label<TransportDriver>(m, class_name);

  py::class_<TransportDriver>(m, class_name.c_str())
    // =================================================================
    // Public constructors
    .def(
      py::init<typename TransportDriver::MeshPtr,
        typename TransportDriver::XSServerPtr,
        const ttnte::parallel::ParallelContext&, std::optional<std::string>>(),
      py::arg("mesh"), py::arg("xs_server"), py::arg("mpi_context"),
      py::arg("label") = std::nullopt)

    // =================================================================
    // Public methods
    .def("distribute", &TransportDriver::distribute,
      "Initial partition using METIS on Rank 0 and culling local mesh.",
      py::arg("load_heuristics") = std::vector<LoadHeuristicPtr> {},
      py::arg("root_rank") = 0)

    .def("redistribute", &TransportDriver::redistribute,
      "Dynamic repartitioning using ParMETIS.",
      py::arg("load_heuristics") = std::vector<LoadHeuristicPtr> {})

    // =================================================================
    // Public getters / setters
    .def_property(
      "label", &TransportDriver::get_label, &TransportDriver::set_label)
    .def_property_readonly("mesh", &TransportDriver::get_mesh)
    .def_property_readonly("server", &TransportDriver::get_server);
}
