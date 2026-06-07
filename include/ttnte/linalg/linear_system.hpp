#pragma once

#include "ttnte/linalg/operator.hpp"
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

  /// Interior operator.
  Operator interior_op_;
  /// State vector.
  State state_;
  /// Source vector.
  State source_;
  /// Whether the state vector is static.
  bool state_is_static_ = false;
  /// Whether the source is static.
  bool source_is_static_ = false;

  // TODO: Once we start looking into the boundary and coupling adjacent patches
  // we will need to include the operators here somehow.

  /// Buffer for the metadata, linear systems, and source.
  torch::Tensor host_buffer_;
  /// Buffer for the metadata, linear systems, and source.
  torch::Tensor device_buffer_;

  /// The current device of the linear system.
  torch::Device device_;

  // =================================================================
  // Protected constructor
  LinearSystem(Operator interior_op, State state = State(),
    State source = State(), std::optional<std::string> label = std::nullopt);

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
  /// @return The interior operator.
  const Operator& get_interior_op() const noexcept { return interior_op_; };
  /// @return The state vector.
  const State& get_state() const noexcept { return state_; }
  /// @return The source vector.
  const State& get_source() const noexcept { return source_; }

  /// @param The new state. Assuming that is not static.
  void set_state(State state);
  /// @param The new source. Assuming that is not static.
  void set_source(State source);

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
};

} // namespace ttnte::linalg
