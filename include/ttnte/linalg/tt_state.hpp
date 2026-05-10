#pragma once

#include "ttnte/linalg/state.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/linalg/tt_operator.hpp"
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
  TTState(
    const TTEngine& tt_vector, std::optional<std::string> label = std::nullopt)
    : State(label.has_value() ? Label::from_string(*label)
                              : Label::create_internal()),
      tt_vector_(tt_vector)
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

  /// @brief In-place left-to-right orthogonalization.
  void lr_orthogonalize_();
  /// @brief Left-to-right orthogonalization.
  Ptr lr_orthogonalize() const;

  /// @brief In-place rounding. Reduce the ranks to a maximum of prescribed
  /// error.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum rank to truncate to.
  void round_(
    double eps = 1e-12, int64_t max_rank = std::numeric_limits<int64_t>::max());
  /// @brief TT-rounding. Reduce the ranks to a maximum of prescribed
  /// error.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum rank to truncate to.
  Ptr round(double eps = 1e-12,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const;

  /// @brief Decompose a tensor into a TT-vector.
  /// @param tensor The tensor to be decomposed.
  /// @return The backend for a TT-vector that represents the original tensor
  /// with some error.
  static Ptr from_dense(const torch::Tensor& tensor, double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max(),
    std::optional<std::string> label = std::nullopt);

  /// @brief Contract rank dimensions to make a full tensor
  torch::Tensor to_dense() const;

  torch::Tensor pack() const override final;
  static Ptr unpack(const torch::Tensor& buffer, bool clone = true);

  /// @brief Flip the sign (in-place) of this TT.
  /// @return The reference to this TT with its sign flipped.
  void neg_();

  /// @brief Create a TT-vector of zeros.
  /// @param m_modes The modes for each core.
  /// @param device The device to put this on.
  /// @param dtype The data type for the cores.
  /// @return The resulting TT-vector.
  static Ptr zeros(const c10::SmallVector<int64_t, 6>& m_modes,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt);
  /// @brief Create a TT-vector of ones.
  /// @param m_modes The modes for each core.
  /// @param device The device to put this on.
  /// @param dtype The data type for the cores.
  /// @return The resulting TT-vector.
  static Ptr ones(const c10::SmallVector<int64_t, 6>& m_modes,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt);

  /// @brief Perform a transpose.
  /// @param cores The indices of which cores to index.
  /// @return The transposed TT.
  TTOperator::Ptr transpose(
    const c10::SmallVector<int64_t, 6>& core_idxs = {}) const;

  // =================================================================
  // Public iterators
  /// @brief Index the vector of TT-cores.
  /// @param i The index position.
  /// @return A constant reference to the core.
  const torch::Tensor& operator[](size_t i) const { return tt_vector_[i]; }
  /// @brief Index the vector of TT-cores.
  /// @param i The index position.
  /// @return A reference to the core.
  torch::Tensor& operator[](size_t i) { return tt_vector_[i]; }

  /// @return Beginning iterator through the TT-cores.
  TTEngine::Tensors::const_iterator begin() const { return tt_vector_.begin(); }
  /// @return Ending iterator through the TT-cores.
  TTEngine::Tensors::const_iterator end() const { return tt_vector_.end(); }

  // =================================================================
  // Public getters / setters
  /// @return The TT-vector backend.
  const TTEngine& get_engine() const noexcept { return tt_vector_; }
  /// @return The number of cores in the TT.
  size_t size() const noexcept { return tt_vector_.size(); }
  /// @return A vector of the TT-cores.
  const TTEngine::Tensors& get_cores() const noexcept
  {
    return tt_vector_.get_cores();
  }
  /// @return Get the ranks of the TT-vector.
  c10::SmallVector<int64_t, 7> get_ranks() const
  {
    return tt_vector_.get_ranks();
  }
  /// @return Get the sizes of the free indices for the TT-vector.
  c10::SmallVector<int64_t, 12> get_free_indices() const
  {
    return tt_vector_.get_free_indices();
  }
  /// @return Get the mode (free index) sizes for this TT-vector.
  c10::SmallVector<int64_t, 6> get_m_modes() const
  {
    return tt_vector_.get_m_modes();
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

// ===================================================================
// Direct sum operators
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTState::Ptr>::type
operator+(const T& a, const TTState::Ptr& b)
{
  return TTState::create(a + b->get_engine());
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTState::Ptr>::type
operator+(const TTState::Ptr& a, const T& b)
{
  return TTState::create(a->get_engine() + b);
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTState::Ptr>::type
operator-(const T& a, const TTState::Ptr& b)
{
  return TTState::create(a - b->get_engine());
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTState::Ptr>::type
operator-(const TTState::Ptr& a, const T& b)
{
  return TTState::create(a->get_engine() - b);
}

inline TTState::Ptr operator+(const TTState::Ptr& a, const TTState::Ptr& b)
{
  return TTState::create(a->get_engine() + b->get_engine());
}
inline TTState::Ptr operator-(const TTState::Ptr& a, const TTState::Ptr& b)
{
  return TTState::create(a->get_engine() - b->get_engine());
}

// ===================================================================
// Hadamard (element-wise) multiplication
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTState::Ptr>::type
operator*(const T& a, const TTState::Ptr& b)
{
  return TTState::create(a * b->get_engine());
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTState::Ptr>::type
operator*(const TTState::Ptr& a, const T& b)
{
  return TTState::create(a->get_engine() * b);
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTState::Ptr>::type
operator/(const T& a, const TTState::Ptr& b)
{
  return TTState::create(a / b->get_engine());
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTState::Ptr>::type
operator/(const TTState::Ptr& a, const T& b)
{
  return TTState::create(a->get_engine() / b);
}

inline TTState::Ptr operator*(const TTState::Ptr& a, const TTState::Ptr& b)
{
  return TTState::create(a->get_engine() * b->get_engine());
}
inline TTState::Ptr operator/(const TTState::Ptr& a, const TTState::Ptr& b)
{
  return TTState::create(a->get_engine() / b->get_engine());
}

} // namespace ttnte::linalg
