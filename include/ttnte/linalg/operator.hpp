#pragma once

#include "ttnte/utils/label.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ttnte::linalg {

/// @brief The abstract class for an operator.
class Operator : public std::enable_shared_from_this<Operator> {
public:
  // =================================================================
  // Public types
  using Label = ttnte::utils::Label<Operator>;
  using Ptr = std::shared_ptr<Operator>;
  using CPtr = std::shared_ptr<const Operator>;

protected:
  // =================================================================
  // Protected constructors
  Operator(std::optional<std::string> label)
    : label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}
  Operator(const Label& label) : label_(label) {}

  // =================================================================
  // Protected data
  /// Label of the operator.
  Label label_;

  // =================================================================
  // Protected methods
  /// @brief Implementation of the `to()` method.
  /// @param device The target device to send the operator to.
  /// @param dtype The new data type to cast the vector's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the operator.
  /// @param memory_format The new memory format of the data.
  /// @return Pointer to the new operator.
  virtual Ptr to_impl(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const = 0;

public:
  virtual ~Operator();

  // =================================================================
  // Public methods
  /// @brief Send or cast the operator.
  /// @param device The target device to send the operator to.
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the operator.
  /// @param memory_format The new memory format of the data.
  /// @return Pointer to the new operator.
  Ptr to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Cast the operator to another type.
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the operator.
  /// @param memory_format The new memory format of the data.
  /// @return Pointer to the new operator.
  Ptr to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Send the operator to another device.
  /// @param device The target device to send the operator to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the operator.
  /// @param memory_format The new memory format of the data.
  /// @return Pointer to the new operator.
  Ptr to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Send or cast the operator (in-place).
  /// @param device The target device to send the operator to.
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the operator.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  /// @brief Cast the operator to another type (in-place).
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the operator.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  /// @brief Send the operator to another device (in-place).
  /// @param device The target device to send the operator to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the operator.
  /// @param memory_format The new memory format of the data.
  virtual void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;

  // =================================================================
  // Public getters / setters
  /// @return Get the label of the operator.
  const Label& get_label() const noexcept { return label_; }
  /// @return Whether the operator is on a CUDA device.
  bool is_cuda() const { return get_device().is_cuda(); }
  /// @return The device the operator is on.
  virtual torch::Device get_device() const = 0;
  /// @return The data type of the operator.
  virtual at::ScalarType get_dtype() const = 0;
  /// @return The storage size of the operator.
  virtual int64_t get_numel() const = 0;
};

} // namespace ttnte::linalg
