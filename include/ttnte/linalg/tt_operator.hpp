#pragma once

#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include <memory>
#include <string>

namespace ttnte::linalg {

/// @brief The TT-matrix class.
class TTOperator : public Operator {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<TTOperator>;

private:
  // =================================================================
  // Private data
  /// The TT-matrix backend.
  TTEngine tt_matrix_;

protected:
  // =================================================================
  // Protected constructors
  TTOperator(const TTEngine::Tensors& cores,
    std::optional<std::string> label = std::nullopt)
    : Operator(label), tt_matrix_(cores)
  {}
  TTOperator(const TTEngine::Tensors& cores, const Label& label)
    : Operator(label), tt_matrix_(cores)
  {}
  TTOperator(const TTEngine& tt_matrix, const Label& label)
    : Operator(label), tt_matrix_(tt_matrix)
  {}

  // =================================================================
  // Protected methods
  /// @brief Implementation of the `to()` method which returns a pointer to the
  /// base class of the new TT-matrix.
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return Shared pointer to the base class of the new TT-matrix.
  Operator::Ptr to_impl(const torch::Device& device,
    const at::ScalarType& dtype, bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) const override final;

public:
  // =================================================================
  // Public methods
  /// Create a shared pointer to a new TT-matrix.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new TTOperator(std::forward<Args>(args)...));
  }
  /// @brief Create a new TT-matrix from a clone of all the provided TT-cores.
  /// @param cores A vector of TT-cores.
  /// @param label The string name of this TT-matrix.
  /// @return Shared pointer to a new TT-matrix.
  static Ptr clone_from(const TTEngine::Tensors& cores,
    std::optional<std::string> label = std::nullopt);
  /// @brief Create a new TT-matrix from a clone of all the provided TT-cores.
  /// @param cores A vector of TT-cores.
  /// @param label The label of this TT-matrix.
  /// @return Shared pointer to a new TT-matrix.
  static Ptr clone_from(const TTEngine::Tensors& cores, const Label& label);

  /// @brief Send the TT-matrix to another device and/or cast it to a different
  /// data type.
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return A shared pointer to the new TT-matrix
  Ptr to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Cast the TT-matrix to a different data type.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return A shared pointer to the new TT-matrix
  Ptr to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Send the TT-matrix to another device.
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return A shared pointer to the new TT-matrix
  Ptr to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Send the TT-matrix to another device and/or cast it to a different
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
  /// @brief Cast the TT-matrix to a different data type (in-place).
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  /// @return A shared pointer to the new TT-matrix
  void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  /// @brief Send the TT-matrix to another device (in-place).
  /// @param device The device to send the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;

  /// @brief Update the TT from a buffer.
  /// @param buffer The buffer of data holding all the TT cores.
  void from_buffer(const torch::Tensor& buffer);

  // =================================================================
  // Public getters / setters
  /// @return The TT-matrix backend.
  const TTEngine& get_engine() const noexcept { return tt_matrix_; }
  /// @return A vector of the TT-cores.
  const TTEngine::Tensors& get_cores() const noexcept
  {
    return tt_matrix_.get_cores();
  }
  /// @return The device that this TT-matrix is on.
  torch::Device get_device() const override final
  {
    return tt_matrix_.get_device();
  }
  /// @return The data type of this TT-matrix.
  at::ScalarType get_dtype() const override final
  {
    return tt_matrix_.get_dtype();
  }
  /// @return The storage size of this TT-matrix.
  int64_t get_numel() const override final { return tt_matrix_.get_numel(); }
};

} // namespace ttnte::linalg
