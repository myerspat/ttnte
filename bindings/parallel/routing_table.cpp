#include "ttnte/parallel/routing_table.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_RoutingTable(py::module_& m)
{
  using namespace ttnte::parallel;

  py::class_<RoutingTable>(m, "RoutingTable")
    .def(py::init<>()) // Default constructor
    .def_readwrite("send_blocks", &RoutingTable::send_blocks,
      "Map of target rank to list of GIDs to send")
    .def_readwrite("recv_blocks", &RoutingTable::recv_blocks,
      "Map of source rank to list of GIDs to receive");
}
