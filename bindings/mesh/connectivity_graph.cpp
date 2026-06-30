#include "ttnte/mesh/connectivity_graph.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_ConnectivityGraph(py::module_& m)
{
  using namespace ttnte::mesh;

  py::class_<ConnectivityGraph>(m, "ConnectivityGraph")
    .def(py::init<>())
    .def_readwrite("local_gids", &ConnectivityGraph::local_gids,
      "Tensor of GIDs for local MeshBlocks")
    .def_readwrite("xadj", &ConnectivityGraph::xadj,
      "CSR-style index pointers for adjacency list")
    .def_readwrite(
      "adjncy", &ConnectivityGraph::adjncy, "Flattened list of neighbor GIDs")
    .def_readwrite("mpi_ranks", &ConnectivityGraph::mpi_ranks,
      "The MPI rank associated with each entry in adjncy");
}
