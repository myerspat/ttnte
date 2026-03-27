#include "ttnte/parallel/communicator.hpp"
#include <mpi.h>

namespace ttnte::parallel {

Communicator Communicator::world()
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int f_comm = MPI_Comm_c2f(MPI_COMM_WORLD);

  return Communicator(f_comm, rank, size);
}

void Communicator::allgather(const int32_t* send_buffer, int send_count,
  int32_t* recv_buffer, int recv_count) const
{
  MPI_Allgather(send_buffer, send_count, MPI_INT32_T, recv_buffer, recv_count,
    MPI_INT32_T, MPI_Comm_f2c(f_comm_));
}

void Communicator::allgather(const int64_t* send_buffer, int send_count,
  int64_t* recv_buffer, int recv_count) const
{
  MPI_Allgather(send_buffer, send_count, MPI_INT64_T, recv_buffer, recv_count,
    MPI_INT64_T, MPI_Comm_f2c(f_comm_));
}

void Communicator::alltoall(const int32_t* send_buffer, int send_count,
  int32_t* recv_buffer, int recv_count) const
{
  // send_count and recv_count are per-rank (usually 1 for this handshake)
  MPI_Alltoall(send_buffer, send_count, MPI_INT32_T, recv_buffer, recv_count,
    MPI_INT32_T, MPI_Comm_f2c(f_comm_));
}

void Communicator::alltoall(const int64_t* send_buffer, int send_count,
  int64_t* recv_buffer, int recv_count) const
{
  // send_count and recv_count are per-rank (usually 1 for this handshake)
  MPI_Alltoall(send_buffer, send_count, MPI_INT64_T, recv_buffer, recv_count,
    MPI_INT64_T, MPI_Comm_f2c(f_comm_));
}

void Communicator::bcast(int64_t* buffer, int count, int root_rank) const
{
  MPI_Bcast(buffer, count, MPI_INT64_T, root_rank, MPI_Comm_f2c(f_comm_));
}

void Communicator::send(
  const int64_t* send_buffer, int count, int target_rank, MPITag tag) const
{
  MPI_Send(send_buffer, count, MPI_INT64_T, target_rank, static_cast<int>(tag),
    MPI_Comm_f2c(f_comm_));
}

void Communicator::recv(
  int64_t* recv_buffer, int count, int source_rank, MPITag tag) const
{
  MPI_Recv(recv_buffer, count, MPI_INT64_T, source_rank, static_cast<int>(tag),
    MPI_Comm_f2c(f_comm_), MPI_STATUS_IGNORE);
}

Request Communicator::isend(
  const int64_t* send_buffer, int count, int target_rank, MPITag tag) const
{
  MPI_Request c_req;
  MPI_Isend(send_buffer, count, MPI_INT64_T, target_rank, static_cast<int>(tag),
    MPI_Comm_f2c(f_comm_), &c_req);
  return Request(MPI_Request_c2f(c_req));
}

Request Communicator::irecv(
  int64_t* recv_buffer, int count, int source_rank, MPITag tag) const
{
  MPI_Request c_req;
  MPI_Irecv(recv_buffer, count, MPI_INT64_T, source_rank, static_cast<int>(tag),
    MPI_Comm_f2c(f_comm_), &c_req);
  return Request(MPI_Request_c2f(c_req));
}

} // namespace ttnte::parallel
