#include "ttnte/parallel/communicator.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_Communicator(py::module_& m)
{
  using namespace ttnte::parallel;

  // Bind the MPITag Enum
  py::enum_<MPITag>(m, "MPITag")
    .value("DEFAULT", MPITag::DEFAULT)
    .value("PARTITION_ID_MAP", MPITag::PARTITION_ID_MAP)
    .value("ROUTING_TABLE", MPITag::ROUTING_TABLE)
    .export_values();

  // Bind the Communicator Class
  py::class_<Communicator>(m, "Communicator")
    .def(py::init<>()) // Default constructor
    .def_static("world", &Communicator::world)

    // Getters
    .def("get", &Communicator::get)
    .def("rank", &Communicator::rank)
    .def("size", &Communicator::size)

    // =================================================================
    // MPI Methods (Using lambdas to cast Python intptr_t to C++ pointers)
    // =================================================================

    .def("allgather",
      [](const Communicator& self, intptr_t send_buf, int send_count,
        intptr_t recv_buf, int recv_count) {
        self.allgather(reinterpret_cast<const int32_t*>(send_buf), send_count,
          reinterpret_cast<int32_t*>(recv_buf), recv_count);
      })

    .def("alltoall",
      [](const Communicator& self, intptr_t send_buf, int send_count,
        intptr_t recv_buf, int recv_count) {
        self.alltoall(reinterpret_cast<const int32_t*>(send_buf), send_count,
          reinterpret_cast<int32_t*>(recv_buf), recv_count);
      })

    .def("bcast",
      [](const Communicator& self, intptr_t buffer, int count, int root_rank) {
        self.bcast(reinterpret_cast<int64_t*>(buffer), count, root_rank);
      })

    .def("send",
      [](const Communicator& self, intptr_t send_buf, int count,
        int target_rank, MPITag tag) {
        self.send(
          reinterpret_cast<const int64_t*>(send_buf), count, target_rank, tag);
      })

    .def("recv",
      [](const Communicator& self, intptr_t recv_buf, int count,
        int source_rank, MPITag tag) {
        self.recv(
          reinterpret_cast<int64_t*>(recv_buf), count, source_rank, tag);
      })

    .def("isend",
      [](const Communicator& self, intptr_t send_buf, int count,
        int target_rank, MPITag tag) {
        return self.isend(
          reinterpret_cast<const int64_t*>(send_buf), count, target_rank, tag);
      })

    .def("irecv", [](const Communicator& self, intptr_t recv_buf, int count,
                    int source_rank, MPITag tag) {
      return self.irecv(
        reinterpret_cast<int64_t*>(recv_buf), count, source_rank, tag);
    });
}
