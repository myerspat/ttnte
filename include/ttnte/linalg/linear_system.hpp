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
  using CPtr = std::shared_ptr<const LinearSystem>;
  using OpPtr = std::shared_ptr<Operator>;
  using StPtr = std::shared_ptr<State>;

protected:
  // =================================================================
  // Protected data
  /// Label of the system.
  Label label_;
  /// Buffer for the metadata, linear systems, and source.
  torch::Tensor host_buffer_;
  /// Buffer for the metadata, linear systems, and source.
  torch::Tensor device_buffer_;

  // State variables
  /// The system has been finalized (is ready for solving).
  bool is_finalized_ = false;

  // =================================================================
  // Protected constructor
  LinearSystem(std::optional<std::string> label = std::nullopt)
    : label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}
  LinearSystem(Label label) : label_(label) {}

  // =================================================================
  // Protected methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::linalg::LinearSystem::" + func_name;
  }

  void is_finalized_or_error(const std::string& func_name) const;

  // virtual Ptr to_impl(const torch::Device& device, const at::ScalarType&
  // dtype,
  //   bool non_blocking = false, bool copy = false,
  //   std::optional<at::MemoryFormat> memory_format = std::nullopt) const = 0;
  // virtual Ptr to_impl(const at::ScalarType& dtype, bool non_blocking = false,
  //   bool copy = false,
  //   std::optional<at::MemoryFormat> memory_format = std::nullopt) const = 0;

public:
  // =================================================================
  // Public methods
  bool is_finalized() const noexcept { return is_finalized_; }

  /// @brief Send or cast the linear system (in-place).
  /// @param device The target device to send the linear system to.
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  /// @brief Cast the linear system to another type (in-place).
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  /// @brief Send the linear system to another device (in-place).
  /// @param device The target device to send the linear system to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;

  // Ptr to(const torch::Device& device, const at::ScalarType& dtype,
  //   bool non_blocking = false, bool copy = false,
  //   std::optional<at::MemoryFormat> memory_format = std::nullopt);
  // Ptr to(const torch::Device& device, bool non_blocking = false,
  //   bool copy = false,
  //   std::optional<at::MemoryFormat> memory_format = std::nullopt);
  // Ptr to(const at::ScalarType& dtype, bool non_blocking = false,
  //   bool copy = false,
  //   std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief Transfer the buffer to another device and data type.
  /// @param device The target device to send the buffer to.
  /// @param dtype The new data type to cast the buffer's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the buffer.
  /// @param memory_format The new memory format of the data.
  virtual void transfer_buffer(const torch::Device& device,
    const at::ScalarType& dtype, bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  /// @brief Transfer the buffer to another device.
  /// @param device The target device to send the buffer to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the buffer.
  /// @param memory_format The new memory format of the data.
  virtual void transfer_buffer(const torch::Device& device,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  /// @brief Transfer the non-buffer data to another device and data type.
  /// @param device The target device to send the non-buffer to.
  /// @param dtype The new data type to cast the non-buffer's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the non-buffer data.
  /// @param memory_format The new memory format of the data.
  virtual void transfer_nonbuffer(const torch::Device& device,
    const at::ScalarType& dtype, bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  /// @brief Transfer the non-buffer data to another device.
  /// @param device The target device to send the non-buffer to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the non-buffer data.
  /// @param memory_format The new memory format of the data.
  virtual void transfer_nonbuffer(const torch::Device& device,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;

  // =================================================================
  // Public getters / setters
  /// @return The label of the linear system.
  const Label& get_label() const noexcept { return label_; }
  /// @return The buffer tensor for static sized data.
  const torch::Tensor& get_buffer(const torch::Device& device)
  {
    return device.is_cuda() ? device_buffer_ : host_buffer_;
  }
  /// @return The interior operator.
  virtual OpPtr get_interior_op() const = 0;
  /// @return The state vector.
  virtual StPtr get_state() const = 0;
  /// @return The source vector.
  virtual StPtr get_source() const = 0;
  /// @return The data type of the linear system.
  at::ScalarType get_dtype() const
  {
    return host_buffer_.defined() ? host_buffer_.scalar_type()
                                  : device_buffer_.scalar_type();
  }

  /// @param The new state vector of the linear system.
  virtual void set_state(const StPtr& new_state) = 0;
};

} // namespace ttnte::linalg
