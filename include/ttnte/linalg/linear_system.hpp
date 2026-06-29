#pragma once

#include "ttnte/linalg/neighbor_coupling.hpp"
#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/source.hpp"
#include "ttnte/linalg/state.hpp"
#include <c10/util/SmallVector.h>
#include <memory>
#include <optional>
#include <string>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/ScalarType.h>

namespace ttnte::linalg {

/// @brief An abstract class for a linear system.
class LinearSystem : public std::enable_shared_from_this<LinearSystem> {
public:
  // =================================================================
  // Public types
  using Label = ttnte::utils::Label<LinearSystem>;
  using Ptr = std::shared_ptr<LinearSystem>;

protected:
  // =================================================================
  // Protected data
  /// Label of the system.
  Label label_;
  /// Global ID of the mesh block this system belongs to. -1 = unset.
  int64_t gid_ = -1;

  /// Interior operator.
  Operator interior_op_;
  /// State vector.
  State state_;
  /// Source object (null if no source has been set).
  Source::Ptr source_;
  /// Whether the state vector is static (packed into the flat buffer).
  bool state_is_static_ = false;

  /// Couplings to neighboring patches at INTERNAL boundary faces.
  c10::SmallVector<NeighborCoupling, 6> couplings_;

  /// Buffer for the metadata, linear systems, and source.
  torch::Tensor host_buffer_;
  /// Buffer for the metadata, linear systems, and source.
  torch::Tensor device_buffer_;

  /// The current device of the linear system.
  torch::Device device_;

  /// Error between the original state and that computed after.
  double error_ = std::numeric_limits<double>::max();

  // =================================================================
  // Protected constructor
  LinearSystem(Operator interior_op,
    c10::SmallVector<NeighborCoupling, 6> couplings = {}, State state = State(),
    Source::Ptr source = nullptr,
    std::optional<std::string> label = std::nullopt);

  // =================================================================
  // Protected methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::linalg::LinearSystem::" + func_name;
  }

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new LinearSystem(std::forward<Args>(args)...));
  }

  /// @brief Transfer the buffer to another device and data type.
  /// @param options The tensor options to apply.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the buffer.
  void transfer_buffer(const torch::TensorOptions& options,
    bool non_blocking = false, bool copy = false);
  /// @brief Transfer the buffer to another device and data type.
  /// @param device The target device to send the buffer to.
  /// @param dtype The new data type to cast the buffer's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the buffer.
  /// @param memory_format The new memory format of the data.
  void transfer_buffer(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  /// @brief Transfer the buffer to another device.
  /// @param device The target device to send the buffer to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the buffer.
  /// @param memory_format The new memory format of the data.
  void transfer_buffer(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief Transfer non-static buffer data to another device or data type.
  /// @param options The tensor options to apply.
  /// @param non_blocking Non-blocking transfer for pinned CPU memory.
  /// @param copy Whether to copy the data.
  void transfer_nonbuffer(const torch::TensorOptions& options,
    bool non_blocking = false, bool copy = false);
  /// @brief Transfer the non-buffer data to another device and data type.
  /// @param device The target device to send the non-buffer to.
  /// @param dtype The new data type to cast the non-buffer's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the non-buffer data.
  /// @param memory_format The new memory format of the data.
  void transfer_nonbuffer(const torch::Device& device,
    const at::ScalarType& dtype, bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  /// @brief Transfer the non-buffer data to another device.
  /// @param device The target device to send the non-buffer to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the non-buffer data.
  /// @param memory_format The new memory format of the data.
  void transfer_nonbuffer(const torch::Device& device,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief Send or cast the linear system (in-place).
  /// @param options The tensor options for all operators and vectors.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  void to_(const torch::TensorOptions& options, bool non_blocking = false,
    bool copy = false);
  /// @brief Send or cast the linear system (in-place).
  /// @param device The target device to send the linear system to.
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  /// @brief Cast the linear system to another type (in-place).
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  /// @brief Send the linear system to another device (in-place).
  /// @param device The target device to send the linear system to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  // =================================================================
  // Public getters / setters
  /// @return The label of the linear system.
  const Label& get_label() const noexcept { return label_; }
  /// @return The global mesh-block ID (-1 if unset).
  int64_t get_gid() const noexcept { return gid_; }
  /// @param gid The global mesh-block ID for this system.
  void set_gid(int64_t gid) noexcept { gid_ = gid; }
  /// @return The couplings to neighboring patches at INTERNAL faces.
  const c10::SmallVector<NeighborCoupling, 6>& get_couplings() const noexcept
  {
    return couplings_;
  }
  /// @return The couplings to neighboring patches at INTERNAL faces.
  c10::SmallVector<NeighborCoupling, 6>& get_couplings() noexcept
  {
    return couplings_;
  }
  /// @return The interior operator.
  const Operator& get_interior_op() const noexcept { return interior_op_; };
  /// @return The state vector.
  const State& get_state() const noexcept { return state_; }
  /// @return The source object (null if no source was provided at
  /// construction).
  const Source::Ptr& get_source() const noexcept { return source_; }

  /// @brief Register a coupling to a neighboring patch at an INTERNAL face.
  /// @param coupling The coupling to add.
  void add_coupling(NeighborCoupling coupling)
  {
    couplings_.push_back(std::move(coupling));
  }

  /// @param state The new state. Assuming that is not static.
  void set_state(State state);

  /// @brief Update the source state from the current solution iterate.
  /// For EigenSource this computes state_ = mv(op_, eigvec) and caches the
  /// sum. No-op for a fixed Source.
  /// @param eigvec Current solution iterate.
  /// @param eps TT rounding tolerance.
  /// @param max_rank Maximum TT rank after rounding.
  void update_source(const State& eigvec, double eps = 1e-12,
    int64_t max_rank = std::numeric_limits<int64_t>::max())
  {
    if (source_)
      source_->update(eigvec, eps, max_rank);
  }

  /// @return The buffer tensor for static sized data.
  const torch::Tensor& get_buffer(const torch::Device& device)
  {
    return device.is_cuda() ? device_buffer_ : host_buffer_;
  }
  /// @return The data type of the linear system.
  at::ScalarType get_dtype() const
  {
    return device_.is_cuda() ? device_buffer_.scalar_type()
                             : host_buffer_.scalar_type();
  }

  /// @return The error of the linear solver.
  double get_error() const noexcept { return error_; }
  /// @param The new error of the linear system.
  void set_error(double error) { error_ = error; }
};

} // namespace ttnte::linalg
