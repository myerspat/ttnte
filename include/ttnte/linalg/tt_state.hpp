#pragma once

#include "ttnte/linalg/state.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include <memory>
#include <string>

namespace ttnte::linalg {

/// @brief The TT-vector class.
class TTState : public State {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<TTState>;

protected:
  // =================================================================
  // Protected data
  /// The TT-vector backend.
  TTEngine tt_vector_;

  // =================================================================
  // Protected constructors
  TTState(const TTEngine::Tensors& cores,
    std::optional<std::string> label = std::nullopt)
    : State(label), tt_vector_(cores)
  {
    check_cores();
  }
  TTState(const TTEngine::Tensors& cores, const Label& label)
    : State(label), tt_vector_(cores)
  {
    check_cores();
  }
  TTState(const TTEngine& tt_vector, const Label& label)
    : State(label), tt_vector_(tt_vector)
  {
    check_cores();
  }

  // =================================================================
  // Protected methods
  /// @brief Implementation of the `to()` method which returns a pointer to the
  /// base class of the new TT-vector.
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return Shared pointer to the base class of the new TT-vector.
  State::Ptr to_impl(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) const override final;

  /// @brief Check the cores are valid.
  void check_cores() const;

public:
  // =================================================================
  // Public methods
  /// Create a shared pointer to a new TT-vector.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new TTState(std::forward<Args>(args)...));
  }
  /// @brief Create a new TT-vector from a clone of all the provided TT-cores.
  /// @param cores A vector of TT-cores.
  /// @param label The string name of this TT-vector.
  /// @return Shared pointer to a new TT-vector.
  static Ptr clone_from(const TTEngine::Tensors& cores,
    std::optional<std::string> label = std::nullopt);
  /// @brief Create a new TT-vector from a clone of all the provided TT-cores.
  /// @param cores A vector of TT-cores.
  /// @param label The label of this TT-vector.
  /// @return Shared pointer to a new TT-vector.
  static Ptr clone_from(const TTEngine::Tensors& cores, const Label& label);

  /// @brief Send the TT-vector to another device and/or cast it to a different
  /// data type.
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return A shared pointer to the new TT-vector
  Ptr to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Cast the TT-vector to a different data type.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return A shared pointer to the new TT-vector
  Ptr to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Send the TT-vector to another device.
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return A shared pointer to the new TT-vector
  Ptr to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Send the TT-vector to another device and/or cast it to a different
  /// data type (in-place).
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  /// @brief Cast the TT-vector to a different data type (in-place).
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return A shared pointer to the new TT-vector
  void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  /// @brief Send the TT-vector to another device (in-place).
  /// @param device The device to send the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;

  // =================================================================
  // Public getters / setters
  /// @return The TT-vector backend.
  const TTEngine& get_engine() const noexcept { return tt_vector_; }
  /// @return A vector of the TT-cores.
  const TTEngine::Tensors& get_cores() const noexcept
  {
    return tt_vector_.get_cores();
  }
  /// @return The device that this TT-vector is on.
  torch::Device get_device() const override final
  {
    return tt_vector_.get_device();
  }
  /// @return The data type of this TT-vector.
  at::ScalarType get_dtype() const override final
  {
    return tt_vector_.get_dtype();
  }
  /// @return The storage size of this TT-vector.
  int64_t get_numel() const override final { return tt_vector_.get_numel(); }
};

} // namespace ttnte::linalg
