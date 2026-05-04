#pragma once

#include "ttnte/utils/label.hpp"
#include <c10/core/Device.h>
#include <c10/util/SmallVector.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ttnte::linalg {

/// @brief The abstract class for a state vector.
class State : public std::enable_shared_from_this<State> {
public:
  // =================================================================
  // Public types
  using Label = ttnte::utils::Label<State>;
  using Ptr = std::shared_ptr<State>;
  using CPtr = std::shared_ptr<const State>;

protected:
  // =================================================================
  // Protected constructors
  State(std::optional<std::string> label)
    : label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}
  State(const Label& label) : label_(label) {}

  // =================================================================
  // Protected data
  /// Label of the state vector.
  Label label_;

  // =================================================================
  // Protected methods
  /// @brief Implementation of the `to()` method.
  /// @param device The target device to send the state vector to.
  /// @param dtype The new data type to cast the vector's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the state vector.
  /// @param memory_format The new memory format of the data.
  /// @return Pointer to the new state vector.
  virtual Ptr to_impl(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const = 0;

public:
  virtual ~State();

  // =================================================================
  // Public methods
  /// @brief Send or cast the state vector.
  /// @param device The target device to send the state vector to.
  /// @param dtype The new data type to cast the vector's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the state vector.
  /// @param memory_format The new memory format of the data.
  /// @return Pointer to the new state vector.
  Ptr to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Cast the state vector to another type.
  /// @param dtype The new data type to cast the vector's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the state vector.
  /// @param memory_format The new memory format of the data.
  /// @return Pointer to the new state vector.
  Ptr to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Send the state vector to another device.
  /// @param device The target device to send the state vector to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the state vector.
  /// @param memory_format The new memory format of the data.
  /// @return Pointer to the new state vector.
  Ptr to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Send or cast the state vector (in-place).
  /// @param device The target device to send the state vector to.
  /// @param dtype The new data type to cast the vector's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the state vector.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  /// @brief Cast the state vector to another type (in-place).
  /// @param dtype The new data type to cast the vector's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the state vector.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  /// @brief Send the state vector to another device (in-place).
  /// @param device The target device to send the state vector to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the state vector.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;

  // =================================================================
  // Public getters / setters
  /// @return The label of the state vector.
  const Label& get_label() const noexcept { return label_; }
  /// @return Whether the state vector is on a CUDA device.
  bool is_cuda() const { return get_device().is_cuda(); }
  /// @return The device the state vector is on.
  virtual torch::Device get_device() const = 0;
  /// @return The data type of the state vector.
  virtual at::ScalarType get_dtype() const = 0;
  /// @return The storage size of the state vector.
  virtual int64_t get_numel() const = 0;
};

} // namespace ttnte::linalg
