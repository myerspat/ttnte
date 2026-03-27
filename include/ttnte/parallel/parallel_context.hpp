#pragma once

#include <torch/extension.h>

namespace ttnte::parallel {

// // Standard types to MPI types
// template<typename T>
// struct MPITypeMap;
//
// template<>
// struct MPITypeMap<int> {
//   static MPI_Datatype value() { return MPI_INT; }
// };
// template<>
// struct MPITypeMap<double> {
//   static MPI_Datatype value() { return MPI_DOUBLE; }
// };
// template<>
// struct MPITypeMap<float> {
//   static MPI_Datatype value() { return MPI_FLOAT; }
// };
// template<>
// struct MPITypeMap<int64_t> {
//   static MPI_Datatype value() { return MPI_LONG_LONG; }
// };

class ParallelContext {
private:
  // =================================================================
  // Private data
  int rank_;
  int world_size_;
  int local_rank_;
  torch::Device device_;
  bool managed_mpi_;

  ParallelContext()
    : rank_(0), world_size_(1), local_rank_(0), device_(torch::kCPU),
      managed_mpi_(false)
  {}

public:
  // Delete copy/move to enforce Singleton
  ParallelContext(const ParallelContext&) = delete;
  void operator=(const ParallelContext&) = delete;

  // =================================================================
  // Public methods
  /// @return Get the instance of the context singleton.
  static ParallelContext& instance();
  /// @brief Initialize the MPI environment.
  void init();
  /// @brief Finalize the MPI environment.
  void finalize();
  /// @brief Create an MPI barrier and stop everything until MPI finishes here.
  void barrier() const;

  // // Send a data buffer across network
  // template<typename T>
  // void inline send(const T* data, int size, int dest, MPITag tag) const
  // {
  //   MPI_Send(data, size, MPITypeMap<T>::value(), dest, static_cast<int>(tag),
  //     MPI_COMM_WORLD);
  // }
  //
  // // Non-blocking version
  // template<typename T>
  // MPI_Request inline isend(const T* data, int size, int dest, MPITag tag)
  // const
  // {
  //   MPI_Request request;
  //   MPI_Isend(data, size, MPITypeMap<T>::value(), dest,
  //   static_cast<int>(tag),
  //     MPI_COMM_WORLD, &request);
  //   return request;
  // }
  //
  // // Pyat versions
  // void inline send(const torch::Tensor& tensor, int dest, MPITag tag) const
  // {
  //   AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "mpi_tensor_send", [&] {
  //     MPI_Send(tensor.contiguous().data_ptr<scalar_t>(), tensor.numel(),
  //       MPITypeMap<scalar_t>::value(), dest, static_cast<int>(tag),
  //       MPI_COMM_WORLD);
  //   });
  // }
  //
  // void inline isend(const torch::Tensor& tensor, int dest, MPITag tag) const
  // {
  //   TORCH_CHECK(tensor.is_contiguous(),
  //     "The tensor of a non-blocking MPI send must be contiguous");
  //
  //   AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "mpi_tensor_send", [&] {
  //     MPI_Send(tensor.data_ptr<scalar_t>(), tensor.numel(),
  //       MPITypeMap<scalar_t>::value(), dest, static_cast<int>(tag),
  //       MPI_COMM_WORLD);
  //   });
  // }
  //
  // // Receive data and store it in a vector
  // template<typename T>
  // std::vector<T> inline recv_vector(int source, MPITag tag) const
  // {
  //   MPI_Status status;
  //   int tag_int = static_cast<int>(tag);
  //
  //   MPI_Probe(source, tag_int, MPI_COMM_WORLD, &status);
  //
  //   int count;
  //   MPI_Get_count(&status, MPITypeMap<T>::value(), &count);
  //
  //   std::vector<T> buffer(count);
  //   MPI_Recv(buffer.data(), count, MPITypeMap<T>::value(), source, tag_int,
  //     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  //   return buffer;
  // }
  //
  // template<typename T>
  // torch::Tensor inline recv_tensor(
  //   int source, MPITag tag, const torch::Device& device = torch::kCPU) const
  // {
  //   MPI_Status status;
  //   int tag_int = static_cast<int>(tag);
  //
  //   MPI_Probe(source, tag_int, MPI_COMM_WORLD, &status);
  //
  //   int count;
  //   MPI_Get_count(&status, MPITypeMap<T>::value(), &count);
  //
  //   // Create receive tensor
  //   torch::Tensor buffer =
  //     torch::empty({count}, torch::TensorOptions().device(device).dtype(
  //                             torch::CppTypeToScalarType<T>::value));
  //
  //   MPI_Recv(buffer.data_ptr<T>(), count, MPITypeMap<T>::value(), source,
  //     tag_int, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  //   return buffer;
  // }

  // =================================================================
  // Public getters / setters
  /// @return The MPI rank.
  int rank() const noexcept { return rank_; }
  /// @return The number of MPI ranks.
  int world_size() const noexcept { return world_size_; }
  /// @return The MPI ranks local to this machine.
  int local_rank() const noexcept { return local_rank_; }
  /// @return The GPU device dedicated to this MPI rank.
  torch::Device device() const noexcept { return device_; }
};

} // namespace ttnte::parallel
