#pragma once

#include <torch/extension.h>

namespace ttnte::linalg {

/// @brief The implementation of all the TT-related functions for both
/// TT-vectors and TT-matrices.
class TTEngine {
public:
  // =================================================================
  // Public types
  using Tensors = c10::SmallVector<torch::Tensor, 6>;

private:
  // =================================================================
  // Private data
  /// The vector of TT-cores.
  Tensors cores_;

public:
  // =================================================================
  // Public constructors
  explicit TTEngine(const Tensors& cores, bool check_cores = true);

  // =================================================================
  // Public methods
  /// @brief Create a new TTEngine from a clone of all the provided TT-cores.
  /// @param cores A vector of TT-cores.
  /// @param check_cores Whether to check if the cores correctly define a TT
  /// object.
  /// @return A new TTEngine object.
  static TTEngine clone_from(const Tensors& cores, bool check_cores = true);
  /// @brief Send the TT to another device and/or cast it to a different data
  /// type.
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  TTEngine to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Cast it to a different data type.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  TTEngine to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  /// @brief Send the TT to another device.
  /// @param device The device to send the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  TTEngine to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Send the TT to another device and/or cast it to a different data
  /// type (in-place).
  /// @param device The device to send the TT to.
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  TTEngine& to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  /// @brief Cast it to a different data type (in-place).
  /// @param dtype The data type to cast the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  TTEngine& to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  /// @brief Send the TT to another device (in-place).
  /// @param device The device to send the TT to.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  /// @param memory_format The new memory format of the data.
  TTEngine& to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief Update the TT from a buffer.
  /// @param buffer The buffer of data holding all the TT cores.
  void from_buffer(const torch::Tensor& buffer);

  // =================================================================
  // Public getters / setters
  /// @return Get the TT-cores.
  const Tensors& get_cores() const noexcept { return cores_; }
  /// @return Get the device of the TT.
  torch::Device get_device() const noexcept { return cores_[0].device(); }
  /// @return Get the data type of the TT.
  at::ScalarType get_dtype() const noexcept { return cores_[0].scalar_type(); }
  /// @return Get the storage size of the TT.
  int64_t get_numel() const;
  /// @return Get the TT-ranks.
  c10::SmallVector<int64_t, 5> get_ranks() const;
  /// @return Get the size of the free indices.
  c10::SmallVector<int64_t, 12> get_free_indices() const;
};

} // namespace ttnte::linalg
