#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/linalg/neighbor_coupling.hpp"
#include "ttnte/linalg/tt_operator.hpp"
#include "ttnte/linalg/tt_state.hpp"

namespace ttnte::linalg {

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
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new TTLinearSystem(std::forward<Args>(args)...));
  }

  void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
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

  void transfer_buffer(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  void transfer_buffer(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  void transfer_nonbuffer(const torch::Device& device,
    const at::ScalarType& dtype, bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  void transfer_nonbuffer(const torch::Device& device,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;

  // =================================================================
  // Public getters / setters
  LinearSystem::OpPtr get_interior_op() const override final
  {
    return interior_op_;
  }
  LinearSystem::StPtr get_state() const override final { return state_; }
  LinearSystem::StPtr get_source() const override final { return source_; }

  void set_state(const LinearSystem::StPtr& new_state) override final
  {
    assert(std::dynamic_pointer_cast<TTState>(new_state));
    state_ = std::static_pointer_cast<TTState>(new_state);
  }
};

} // namespace ttnte::linalg
