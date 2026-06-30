#include "ttnte/parallel/boundary_communicator.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_BoundaryCommunicator(py::module_& m)
{
  using namespace ttnte::parallel;

  py::class_<BoundaryCommunicator>(m, "BoundaryCommunicator")
    // =================================================================
    // Public Constructor
    .def(py::init<const Communicator&, int64_t>(), py::arg("world_comm"),
      py::arg("num_boundaries"),
      "Constructs a set of duplicated MPI communicators for domain boundaries.")

    // =================================================================
    // Public Static Methods
    .def_static("generate_tag", &BoundaryCommunicator::generate_tag,
      py::arg("sender_id"), py::arg("sender_level"), py::arg("recv_id"),
      py::arg("recv_level"),
      "Generates a deterministic MPI tag based on sender/receiver topology.")

    // =================================================================
    // Public Methods & Getters
    .def("get_comm", &BoundaryCommunicator::get_comm, py::arg("fid"),
      py::return_value_policy::reference_internal,
      "Returns the duplicated Communicator for a specific face ID.")
    .def(
      "get_comms",
      [](const BoundaryCommunicator& self) {
        py::list l;
        for (const auto& comm : self.get_comms()) {
          // Return by reference so we don't trigger the deleted copy
          // constructor
          l.append(py::cast(comm, py::return_value_policy::reference_internal));
        }
        return l;
      },
      "Returns a list of all boundary communicators.");
}
