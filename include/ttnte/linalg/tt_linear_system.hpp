#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/linalg/neighbor_coupling.hpp"
#include "ttnte/linalg/tt_operator.hpp"
#include "ttnte/linalg/tt_state.hpp"

namespace ttnte::linalg {

/// @brief A class for a linear system in TT format.
class TTLinearSystem : public LinearSystem {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<TTLinearSystem>;
  using CPtr = std::shared_ptr<const TTLinearSystem>;
  using OpPtr = std::shared_ptr<TTOperator>;
  using StPtr = std::shared_ptr<TTState>;
  using BoundaryConnections = c10::SmallVector<NeighborCoupling<TTOperator>, 6>;

protected:
  // =================================================================
  // Protected data
  /// Interior operator.
  OpPtr interior_op_;
  /// Solution vector.
  StPtr state_;
  /// Source vector.
  StPtr source_;

  // =================================================================
  // Protected constructor
  TTLinearSystem(std::optional<std::string> label = std::nullopt)
    : LinearSystem(label)
  {}
  TTLinearSystem(Label label) : LinearSystem(label) {}
  TTLinearSystem(const OpPtr& interior_op, const StPtr& state,
    const StPtr& source, std::optional<std::string> label = std::nullopt);

  // =================================================================
  // Protected methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::linalg::TTLinearSystem::" + func_name;
  }

  // LinearSystem::Ptr to_impl(const torch::Device& device,
  //   const at::ScalarType& dtype, bool non_blocking = false, bool copy =
  //   false, std::optional<at::MemoryFormat> memory_format =
  //     std::nullopt) const override final;
  // LinearSystem::Ptr to_impl(const at::ScalarType& dtype,
  //   bool non_blocking = false, bool copy = false,
  //   std::optional<at::MemoryFormat> memory_format =
  //     std::nullopt) const override final;

public:
  // =================================================================
  // Public methods
  /// @brief Construct a shared pointer to a linear system in TT format.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new TTLinearSystem(std::forward<Args>(args)...));
  }

  /// @brief Send or cast the linear system (in-place).
  /// @param device The target device to send the linear system to.
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  /// @brief Cast the linear system to another type (in-place).
  /// @param dtype The new data type to cast the data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  /// @brief Send the linear system to another device (in-place).
  /// @param device The target device to send the linear system to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the linear system.
  /// @param memory_format The new memory format of the data.
  void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;

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
  void transfer_buffer(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  /// @brief Transfer the buffer to another device.
  /// @param device The target device to send the buffer to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the buffer.
  /// @param memory_format The new memory format of the data.
  void transfer_buffer(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  /// @brief Transfer the non-buffer data to another device and data type.
  /// @param device The target device to send the non-buffer to.
  /// @param dtype The new data type to cast the non-buffer's data to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the non-buffer data.
  /// @param memory_format The new memory format of the data.
  void transfer_nonbuffer(const torch::Device& device,
    const at::ScalarType& dtype, bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  /// @brief Transfer the non-buffer data to another device.
  /// @param device The target device to send the non-buffer to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the non-buffer data.
  /// @param memory_format The new memory format of the data.
  void transfer_nonbuffer(const torch::Device& device,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;

  // =================================================================
  // Public getters / setters
  /// @return The interior operator.
  LinearSystem::OpPtr get_interior_op() const override final
  {
    return interior_op_;
  }
  /// @return The state vector.
  LinearSystem::StPtr get_state() const override final { return state_; }
  /// @return The source vector.
  LinearSystem::StPtr get_source() const override final { return source_; }

  /// @param The new state vector of the linear system.
  void set_state(const LinearSystem::StPtr& new_state) override final
  {
    assert(std::dynamic_pointer_cast<TTState>(new_state));
    state_ = std::static_pointer_cast<TTState>(new_state);
  }
};

} // namespace ttnte::linalg
