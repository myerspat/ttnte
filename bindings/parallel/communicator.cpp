#include "ttnte/parallel/communicator.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_Communicator(py::module_& m)
{
  using namespace ttnte::parallel;

  // Bind the Enums
  py::enum_<MPITag>(m, "MPITag")
    .value("DEFAULT", MPITag::DEFAULT)
    .value("PARTITION_ID_MAP", MPITag::PARTITION_ID_MAP)
    .value("ROUTING_TABLE", MPITag::ROUTING_TABLE)
    .export_values();

  py::enum_<DataType>(m, "DataType")
    .value("INT32", DataType::INT32)
    .value("INT64", DataType::INT64)
    .value("FLOAT", DataType::FLOAT)
    .value("DOUBLE", DataType::DOUBLE)
    .value("BYTE", DataType::BYTE)
    .export_values();

  // Bind ProbeResult
  py::class_<ProbeResult>(m, "ProbeResult")
    .def_readwrite("matched", &ProbeResult::matched)
    .def_readwrite("count", &ProbeResult::count)
    .def_readwrite("message_f", &ProbeResult::message_f);

  // Bind the Communicator Class
  py::class_<Communicator>(m, "Communicator")
    .def(py::init<>()) // Default constructor
    .def_static("world", &Communicator::world)
    .def("duplicate", &Communicator::duplicate)
    .def("is_world_comm", &Communicator::is_world_comm)

    // Getters
    .def("get", &Communicator::get)
    .def("rank", &Communicator::rank)
    .def("size", &Communicator::size)

    // =================================================================
    // MPI Methods (Dynamically dispatched via the DataType enum)
    // =================================================================
    .def("allgather",
      [](const Communicator& self, intptr_t send_buf, int send_count,
        intptr_t recv_buf, int recv_count, DataType dtype) {
        TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.allgather(reinterpret_cast<const T*>(send_buf), send_count,
            reinterpret_cast<T*>(recv_buf), recv_count));
      })

    .def("alltoall",
      [](const Communicator& self, intptr_t send_buf, int send_count,
        intptr_t recv_buf, int recv_count, DataType dtype) {
        TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.alltoall(reinterpret_cast<const T*>(send_buf), send_count,
            reinterpret_cast<T*>(recv_buf), recv_count));
      })

    .def("bcast",
      [](const Communicator& self, intptr_t buffer, int count, int root_rank,
        DataType dtype) {
        TTNTE_DISPATCH_DATATYPE(
          dtype, T, self.bcast(reinterpret_cast<T*>(buffer), count, root_rank));
      })
    .def("iallgather",
      [](const Communicator& self, intptr_t send_buf, int send_count,
        intptr_t recv_buf, int recv_count, DataType dtype) {
        return TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.iallgather(reinterpret_cast<const T*>(send_buf), send_count,
            reinterpret_cast<T*>(recv_buf), recv_count));
      })
    .def("imrecv",
      [](const Communicator& self, intptr_t buffer, int count, DataType type,
        ProbeResult& probe) {
        return self.imrecv(reinterpret_cast<void*>(buffer), count, type, probe);
      })

    // MPITag
    .def("send",
      [](const Communicator& self, intptr_t send_buf, int count,
        int target_rank, MPITag tag, DataType dtype) {
        TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.send(
            reinterpret_cast<const T*>(send_buf), count, target_rank, tag));
      })
    .def("recv",
      [](const Communicator& self, intptr_t recv_buf, int count,
        int source_rank, MPITag tag, DataType dtype) {
        TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.recv(reinterpret_cast<T*>(recv_buf), count, source_rank, tag));
      })
    .def("isend",
      [](const Communicator& self, intptr_t send_buf, int count,
        int target_rank, MPITag tag, DataType dtype) {
        return TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.isend(
            reinterpret_cast<const T*>(send_buf), count, target_rank, tag));
      })
    .def("irecv",
      [](const Communicator& self, intptr_t recv_buf, int count,
        int source_rank, MPITag tag, DataType dtype) {
        return TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.irecv(reinterpret_cast<T*>(recv_buf), count, source_rank, tag));
      })
    .def(
      "iprobe", [](const Communicator& self, int source, MPITag tag,
                  DataType dtype) { return self.iprobe(source, tag, dtype); })

    // int
    .def("send",
      [](const Communicator& self, intptr_t send_buf, int count,
        int target_rank, int tag, DataType dtype) {
        TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.send(
            reinterpret_cast<const T*>(send_buf), count, target_rank, tag));
      })
    .def("recv",
      [](const Communicator& self, intptr_t recv_buf, int count,
        int source_rank, int tag, DataType dtype) {
        TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.recv(reinterpret_cast<T*>(recv_buf), count, source_rank, tag));
      })
    .def("isend",
      [](const Communicator& self, intptr_t send_buf, int count,
        int target_rank, int tag, DataType dtype) {
        return TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.isend(
            reinterpret_cast<const T*>(send_buf), count, target_rank, tag));
      })
    .def("irecv",
      [](const Communicator& self, intptr_t recv_buf, int count,
        int source_rank, int tag, DataType dtype) {
        return TTNTE_DISPATCH_DATATYPE(dtype, T,
          self.irecv(reinterpret_cast<T*>(recv_buf), count, source_rank, tag));
      })

    .def(
      "iprobe", [](const Communicator& self, int source, int tag,
                  DataType dtype) { return self.iprobe(source, tag, dtype); });
}
