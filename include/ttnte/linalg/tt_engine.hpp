#pragma once

#include <torch/extension.h>

namespace ttnte::linalg {

class TTEngine {
public:
  // =================================================================
  // Public types
  using Tensors = c10::SmallVector<torch::Tensor, 6>;

private:
  // =================================================================
  // Private data
  Tensors cores_;

public:
  // =================================================================
  // Public constructors
  explicit TTEngine(const Tensors& cores, bool check_cores = true);

  // =================================================================
  // Public methods
  static TTEngine clone_from(const Tensors& cores, bool check_cores = true);
  TTEngine to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  TTEngine to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  TTEngine to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  TTEngine& to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  TTEngine& to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);
  TTEngine& to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  void from_buffer(const torch::Tensor& buffer);

  // =================================================================
  // Public getters / setters
  const Tensors& get_cores() const noexcept { return cores_; }
  torch::Device get_device() const noexcept { return cores_[0].device(); }
  at::ScalarType get_dtype() const noexcept { return cores_[0].scalar_type(); }
  int64_t get_numel() const;
  c10::SmallVector<int64_t, 5> get_ranks() const;
  c10::SmallVector<int64_t, 12> get_free_indices() const;
};

} // namespace ttnte::linalg
