#pragma once

#include "ttnte/utils/label.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ttnte::linalg {

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
  Label label_;

  // =================================================================
  // Protected methods
  virtual Ptr to_impl(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const = 0;

public:
  virtual ~Operator();

  // =================================================================
  // Public methods
  Ptr to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  Ptr to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  Ptr to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  virtual void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  virtual void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;
  virtual void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) = 0;

  // =================================================================
  // Public getters / setters
  const Label& get_label() const noexcept { return label_; }
  bool is_cuda() const { return get_device().is_cuda(); }
  virtual torch::Device get_device() const = 0;
  virtual at::ScalarType get_dtype() const = 0;
  virtual int64_t get_numel() const = 0;
};

} // namespace ttnte::linalg
