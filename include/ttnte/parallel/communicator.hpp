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

enum class DataType : int8_t { INT32, INT64, FLOAT, DOUBLE, BYTE };

struct ProbeResult {
  bool matched;
  int count;
  int message_f;
};

#define TTNTE_DISPATCH_DATATYPE(dtype, TYPE_VAR, ...)                          \
  [&] {                                                                        \
    switch (dtype) {                                                           \
    case ttnte::parallel::DataType::INT32: {                                   \
      using TYPE_VAR = int32_t;                                                \
      return __VA_ARGS__;                                                      \
    }                                                                          \
    case ttnte::parallel::DataType::INT64: {                                   \
      using TYPE_VAR = int64_t;                                                \
      return __VA_ARGS__;                                                      \
    }                                                                          \
    case ttnte::parallel::DataType::FLOAT: {                                   \
      using TYPE_VAR = float;                                                  \
      return __VA_ARGS__;                                                      \
    }                                                                          \
    case ttnte::parallel::DataType::DOUBLE: {                                  \
      using TYPE_VAR = double;                                                 \
      return __VA_ARGS__;                                                      \
    }                                                                          \
    case ttnte::parallel::DataType::BYTE: {                                    \
      using TYPE_VAR = std::byte;                                              \
      return __VA_ARGS__;                                                      \
    }                                                                          \
    default:                                                                   \
      throw std::runtime_error("Unsupported DataType");                        \
    }                                                                          \
  }()

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
  /// State variable for whether this is a duplicated communicator.
  bool is_dynamic_;

  // =================================================================
  // Private constructor
  Communicator(int f_comm, int rank, int size, bool is_dynamic)
    : f_comm_(f_comm), rank_(rank), size_(size), is_dynamic_(is_dynamic)
  {}

  // =================================================================
  // Private methods
  int duplicate_internal() const;
  void free_internal();

public:
  // =================================================================
  // Public constructors
  // Default constructor (creates a null/invalid communicator)
  Communicator() : f_comm_(0), rank_(-1), size_(-1), is_dynamic_(false) {}
  // Enable moving
  Communicator(Communicator&& other) noexcept
    : f_comm_(other.f_comm_), rank_(other.rank_), size_(other.size_),
      is_dynamic_(other.is_dynamic_)
  {
    other.is_dynamic_ = false; // Steal ownership
    other.f_comm_ = 0;
  }

  /// Destructor that frees the internal Comm (only for duplicates).
  ~Communicator()
  {
    if (is_dynamic_ && f_comm_ != 0) {
      free_internal();
    }
  }

  // Disable copying (prevents double-free crashes)
  Communicator(const Communicator&) = delete;
  Communicator& operator=(const Communicator&) = delete;

  // Factory methods
  /// @return The base communicator for this MPI rank.
  static Communicator world();
  /// @return A duplicate of the base communicator.
  Communicator duplicate() const;

  // =================================================================
  // Public operators
  /// Assignment operator
  Communicator& operator=(Communicator&& other) noexcept
  {
    if (this != &other) {
      if (is_dynamic_)
        free_internal(); // free current resource
      f_comm_ = other.f_comm_;
      rank_ = other.rank_;
      size_ = other.size_;
      is_dynamic_ = other.is_dynamic_;
      other.is_dynamic_ = false;
      other.f_comm_ = 0;
    }
    return *this;
  }

  // =================================================================
  // Public methods
  // Memory management methods
  bool is_world_comm() const;

  template<typename BufferType>
  void allgather(const BufferType* send_buffer, int send_count,
    BufferType* recv_buffer, int recv_count) const;

  template<typename BufferType>
  void alltoall(const BufferType* send_buffer, int send_count,
    BufferType* recv_buffer, int recv_count) const;

  template<typename BufferType>
  void bcast(BufferType* buffer, int count, int root_rank) const;

  template<typename BufferType, typename Tag>
  void send(
    const BufferType* send_buffer, int count, int target_rank, Tag tag) const;

  template<typename BufferType, typename Tag>
  void recv(BufferType* recv_buffer, int count, int source_rank, Tag tag) const;

  template<typename BufferType>
  Request iallgather(const BufferType* send_buffer, int send_count,
    BufferType* recv_buffer, int recv_count) const;

  template<typename BufferType, typename Tag>
  Request isend(
    const BufferType* send_buffer, int count, int target_rank, Tag tag) const;

  template<typename BufferType, typename Tag>
  Request irecv(
    BufferType* recv_buffer, int count, int source_rank, Tag tag) const;

  template<typename Tag>
  ProbeResult iprobe(int source, Tag tag, DataType type) const;

  Request imrecv(
    void* buffer, int count, DataType type, ProbeResult& probe) const;

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
