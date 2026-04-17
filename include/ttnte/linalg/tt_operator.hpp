#pragma once

#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include <memory>
#include <string>

namespace ttnte::linalg {

class TTOperator : public Operator {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<TTOperator>;

private:
  // =================================================================
  // Private data
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
  Operator::Ptr to_impl(const torch::Device& device,
    const at::ScalarType& dtype, bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) const override final;

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new TTOperator(std::forward<Args>(args)...));
  }
  static Ptr clone_from(const TTEngine::Tensors& cores,
    std::optional<std::string> label = std::nullopt);
  static Ptr clone_from(const TTEngine::Tensors& cores, const Label& label);

  Ptr to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  Ptr to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;
  Ptr to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  void to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  void to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;
  void to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) override final;

  void from_buffer(const torch::Tensor& buffer);

  // =================================================================
  // Public getters / setters
  const TTEngine& get_engine() const noexcept { return tt_matrix_; }
  const TTEngine::Tensors& get_cores() const noexcept
  {
    return tt_matrix_.get_cores();
  }
  torch::Device get_device() const override final
  {
    return tt_matrix_.get_device();
  }
  at::ScalarType get_dtype() const override final
  {
    return tt_matrix_.get_dtype();
  }
  int64_t get_numel() const override final { return tt_matrix_.get_numel(); }
};

} // namespace ttnte::linalg
