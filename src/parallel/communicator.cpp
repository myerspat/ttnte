#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/mpi_types.hpp"
#include <cstddef>
#include <mpi.h>

namespace {

static MPI_Datatype ttnte2mpi(ttnte::parallel::DataType type)
{
  using DataType = ttnte::parallel::DataType;

  switch (type) {
  case DataType::INT32:
    return MPI_INT32_T;
  case DataType::INT64:
    return MPI_INT64_T;
  case DataType::FLOAT:
    return MPI_FLOAT;
  case DataType::DOUBLE:
    return MPI_DOUBLE;
  case DataType::BYTE:
    return MPI_BYTE;
  default:
    return MPI_DATATYPE_NULL;
  }
}

} // namespace

namespace ttnte::parallel {

Communicator Communicator::world()
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int f_comm = MPI_Comm_c2f(MPI_COMM_WORLD);

  return Communicator(f_comm, rank, size, false);
}

bool Communicator::is_world_comm() const
{
  return f_comm_ == MPI_Comm_c2f(MPI_COMM_WORLD);
}

Communicator Communicator::duplicate() const
{
  // Dup creates a new dynamic comm
  int new_comm = duplicate_internal();
  return Communicator(new_comm, rank_, size_, true);
}

int Communicator::duplicate_internal() const
{
  // Convert our type-erased int handle back to a C MPI_Comm
  MPI_Comm c_comm = MPI_Comm_f2c(f_comm_);
  MPI_Comm new_c_comm;

  // Tell MPI to duplicate the communicator
  // This creates a new 'context' with the same ranks
  MPI_Comm_dup(c_comm, &new_c_comm);

  // Convert the new C communicator back to an int handle for our class
  return MPI_Comm_c2f(new_c_comm);
}

void Communicator::free_internal()
{
  // 1. Convert back to C type
  MPI_Comm c_comm = MPI_Comm_f2c(f_comm_);

  // 2. Safety check: Never free the null communicator
  if (c_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&c_comm);
  }

  // 3. Update our internal handle to the "Null" state
  f_comm_ = MPI_Comm_c2f(MPI_COMM_NULL);
}

template<typename BufferType>
void Communicator::allgather(const BufferType* send_buffer, int send_count,
  BufferType* recv_buffer, int recv_count) const
{
  // Get the MPI type
  auto type = MPI_Type_f2c(ttnte::parallel::get_mpi_type<BufferType>());

  MPI_Allgather(send_buffer, send_count, type, recv_buffer, recv_count, type,
    MPI_Comm_f2c(f_comm_));
}

template<typename BufferType>
void Communicator::alltoall(const BufferType* send_buffer, int send_count,
  BufferType* recv_buffer, int recv_count) const
{
  // Get the MPI type
  auto type = MPI_Type_f2c(ttnte::parallel::get_mpi_type<BufferType>());

  // send_count and recv_count are per-rank (usually 1 for this handshake)
  MPI_Alltoall(send_buffer, send_count, type, recv_buffer, recv_count, type,
    MPI_Comm_f2c(f_comm_));
}

template<typename BufferType>
void Communicator::bcast(BufferType* buffer, int count, int root_rank) const
{
  // Get the MPI type
  auto type = MPI_Type_f2c(ttnte::parallel::get_mpi_type<BufferType>());

  MPI_Bcast(buffer, count, type, root_rank, MPI_Comm_f2c(f_comm_));
}

template<typename BufferType, typename Tag>
void Communicator::send(
  const BufferType* send_buffer, int count, int target_rank, Tag tag) const
{
  // Get the MPI type
  auto type = MPI_Type_f2c(ttnte::parallel::get_mpi_type<BufferType>());

  MPI_Send(send_buffer, count, type, target_rank, static_cast<int>(tag),
    MPI_Comm_f2c(f_comm_));
}

template<typename BufferType, typename Tag>
void Communicator::recv(
  BufferType* recv_buffer, int count, int source_rank, Tag tag) const
{
  // Get the MPI type
  auto type = MPI_Type_f2c(ttnte::parallel::get_mpi_type<BufferType>());

  MPI_Recv(recv_buffer, count, type, source_rank, static_cast<int>(tag),
    MPI_Comm_f2c(f_comm_), MPI_STATUS_IGNORE);
}

template<typename BufferType>
Request Communicator::iallgather(const BufferType* send_buffer, int send_count,
  BufferType* recv_buffer, int recv_count) const
{
  // Get the MPI type
  auto type = MPI_Type_f2c(ttnte::parallel::get_mpi_type<BufferType>());

  MPI_Request c_req;
  MPI_Iallgather(send_buffer, send_count, type, recv_buffer, recv_count, type,
    MPI_Comm_f2c(f_comm_), &c_req);
  return Request(MPI_Request_c2f(c_req));
}

template<typename BufferType, typename Tag>
Request Communicator::isend(
  const BufferType* send_buffer, int count, int target_rank, Tag tag) const
{
  // Get the MPI type
  auto type = MPI_Type_f2c(ttnte::parallel::get_mpi_type<BufferType>());

  MPI_Request c_req;
  MPI_Isend(send_buffer, count, type, target_rank, static_cast<int>(tag),
    MPI_Comm_f2c(f_comm_), &c_req);
  return Request(MPI_Request_c2f(c_req));
}

template<typename BufferType, typename Tag>
Request Communicator::irecv(
  BufferType* recv_buffer, int count, int source_rank, Tag tag) const
{
  // Get the MPI type
  auto type = MPI_Type_f2c(ttnte::parallel::get_mpi_type<BufferType>());

  MPI_Request c_req;
  MPI_Irecv(recv_buffer, count, type, source_rank, static_cast<int>(tag),
    MPI_Comm_f2c(f_comm_), &c_req);
  return Request(MPI_Request_c2f(c_req));
}

template<typename Tag>
ProbeResult Communicator::iprobe(int source, Tag tag, DataType type) const
{
  MPI_Message msg;
  MPI_Status status;
  int flag = 0;

  MPI_Improbe(
    source, static_cast<int>(tag), MPI_Comm_f2c(f_comm_), &flag, &msg, &status);

  if (!flag)
    return {false, 0, 0};

  int count;
  // Returns number of elements of 'mpi_type'
  MPI_Get_count(&status, ttnte2mpi(type), &count);
  return {true, count, MPI_Message_c2f(msg)};
}

Request Communicator::imrecv(
  void* buffer, int count, DataType type, ProbeResult& probe) const
{
  MPI_Request req;
  MPI_Message msg = MPI_Message_f2c(probe.message_f);

  MPI_Imrecv(buffer, count, ttnte2mpi(type), &msg, &req);
  return Request(MPI_Request_c2f(req));
}

// --- Collective Operations (Single Template Parameter) ---

#define INSTANTIATE_COLLECTIVE(Type)                                           \
  template void Communicator::allgather<Type>(const Type* send_buffer,         \
    int send_count, Type* recv_buffer, int recv_count) const;                  \
  template Request Communicator::iallgather<Type>(const Type* send_buffer,     \
    int send_count, Type* recv_buffer, int recv_count) const;                  \
  template void Communicator::alltoall<Type>(const Type* send_buffer,          \
    int send_count, Type* recv_buffer, int recv_count) const;                  \
  template void Communicator::bcast<Type>(                                     \
    Type * buffer, int count, int root_rank) const;

INSTANTIATE_COLLECTIVE(std::byte)
INSTANTIATE_COLLECTIVE(int32_t)
INSTANTIATE_COLLECTIVE(int64_t)
INSTANTIATE_COLLECTIVE(float)
INSTANTIATE_COLLECTIVE(double)

// --- Point-to-Point Operations (Double Template Parameter) ---
// We must instantiate every combination of BufferType + {MPITag, int}

#define INSTANTIATE_P2P(Buffer, TagType)                                       \
  template void Communicator::send<Buffer, TagType>(                           \
    const Buffer*, int, int, TagType) const;                                   \
  template void Communicator::recv<Buffer, TagType>(                           \
    Buffer*, int, int, TagType) const;                                         \
  template Request Communicator::isend<Buffer, TagType>(                       \
    const Buffer*, int, int, TagType) const;                                   \
  template Request Communicator::irecv<Buffer, TagType>(                       \
    Buffer*, int, int, TagType) const;

// Instantiations for MPITag
INSTANTIATE_P2P(std::byte, MPITag)
INSTANTIATE_P2P(int32_t, MPITag)
INSTANTIATE_P2P(int64_t, MPITag)
INSTANTIATE_P2P(float, MPITag)
INSTANTIATE_P2P(double, MPITag)

// Instantiations for raw int tags
INSTANTIATE_P2P(std::byte, int)
INSTANTIATE_P2P(int32_t, int)
INSTANTIATE_P2P(int64_t, int)
INSTANTIATE_P2P(float, int)
INSTANTIATE_P2P(double, int)

// --- Probe Operations (Tag Template Parameter) ---

template ProbeResult Communicator::iprobe<MPITag>(int, MPITag, DataType) const;
template ProbeResult Communicator::iprobe<int>(int, int, DataType) const;

} // namespace ttnte::parallel
