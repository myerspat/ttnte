#pragma once

#include "ttnte/linalg/state.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include <memory>
#include <string>

namespace ttnte::linalg {

class TTState : public State {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<TTState>;

protected:
  // =================================================================
  // Protected data
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
  State::Ptr to_impl(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format =
      std::nullopt) const override final;

  void check_cores() const;

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new TTState(std::forward<Args>(args)...));
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

  // =================================================================
  // Public getters / setters
  const TTEngine& get_engine() const noexcept { return tt_vector_; }
  const TTEngine::Tensors& get_cores() const noexcept
  {
    return tt_vector_.get_cores();
  }
  torch::Device get_device() const override final
  {
    return tt_vector_.get_device();
  }
  at::ScalarType get_dtype() const override final
  {
    return tt_vector_.get_dtype();
  }
  int64_t get_numel() const override final { return tt_vector_.get_numel(); }
};

} // namespace ttnte::linalg
