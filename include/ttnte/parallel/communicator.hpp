#pragma once

#include "ttnte/parallel/request.hpp"
#include <cstdint>

namespace ttnte::parallel {

/// @brief Tags for MPI communication.
enum class MPITag : int {
  DEFAULT = 0,

  // Load balancing
  PARTITION_ID_MAP = 101,
  ROUTING_TABLE = 102,
};

/// @brief The communicator handles MPI related communication outside of the
/// task-based system.
class Communicator {
private:
  // =================================================================
  // Private data
  /// Type erased MPI_COMM_WORLD.
  int f_comm_;
  /// The MPI_rank.
  int rank_;
  /// The number of MPI ranks.
  int size_;

  // =================================================================
  // Private constructor
  Communicator(int f_comm, int rank, int size)
    : f_comm_(f_comm), rank_(rank), size_(size)
  {}

public:
  // =================================================================
  // Public constructors
  // Default constructor (creates a null/invalid communicator)
  Communicator() : f_comm_(0), rank_(-1), size_(-1) {}

  // Factory methods
  static Communicator world();

  // =================================================================
  // Public methods
  void allgather(const int32_t* send_buffer, int send_count,
    int32_t* recv_buffer, int recv_count) const;
  void allgather(const int64_t* send_buffer, int send_count,
    int64_t* recv_buffer, int recv_count) const;
  void alltoall(const int32_t* send_buffer, int send_count,
    int32_t* recv_buffer, int recv_count) const;
  void alltoall(const int64_t* send_buffer, int send_count,
    int64_t* recv_buffer, int recv_count) const;

  void bcast(int64_t* buffer, int count, int root_rank) const;

  void send(
    const int64_t* send_buffer, int count, int target_rank, MPITag tag) const;
  void recv(int64_t* recv_buffer, int count, int source_rank, MPITag tag) const;

  Request isend(
    const int64_t* send_buffer, int count, int target_rank, MPITag tag) const;
  Request irecv(
    int64_t* recv_buffer, int count, int source_rank, MPITag tag) const;

  // =================================================================
  // Public getters
  /// @return Get the MPI_COMM_WORLD as a typed erased integer.
  int get() const { return f_comm_; }
  /// @return Get the MPI rank.
  int rank() const { return rank_; }
  /// @return Get the number of MPI ranks.
  int size() const { return size_; }
};

} // namespace ttnte::parallel
