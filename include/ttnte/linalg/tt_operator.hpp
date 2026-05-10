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
  TTOperator(
    const TTEngine& tt_matrix, std::optional<std::string> label = std::nullopt)
    : Operator(label.has_value() ? Label::from_string(*label)
                                 : Label::create_internal()),
      tt_matrix_(tt_matrix)
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

  /// @brief In-place left-to-right orthogonalization.
  void lr_orthogonalize_();
  /// @brief Left-to-right orthogonalization.
  Ptr lr_orthogonalize();

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
  Ptr round(
    double eps = 1e-12, int64_t max_rank = std::numeric_limits<int64_t>::max());

  /// @brief Decompose a tensor into a TT-matrix with the given shape.
  /// @param m_modes The output modes for each core.
  /// @param n_modes The input modes for each core.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum rank to truncate to.
  /// @param is_interleaved Whether the tensor is already permuted to
  /// (n_0, m_0, n_1, m_1, ...) or is (n_0, n_1, ..., m_0, m_1, ...)
  /// @return A TT-matrix that represents the original tensor
  /// with some error.
  static Ptr from_dense(torch::Tensor tensor,
    const c10::SmallVector<int64_t, 6>& m_modes,
    const c10::SmallVector<int64_t, 6>& n_modes, double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max(),
    bool is_interleaved = false,
    std::optional<std::string> label = std::nullopt);

  /// @brief Contract rank dimensions to make a full tensor
  torch::Tensor to_dense(bool interleave = false) const;

  /// @brief Update the TT from a buffer.
  /// @param buffer The buffer of data holding all the TT cores.
  void from_buffer(const torch::Tensor& buffer);

  /// @brief Flip the sign (in-place) of this TT.
  /// @return The reference to this TT with its sign flipped.
  void neg_();

  // Factory methods
  /// @brief Create a TT-matrix of zeros.
  /// @param m_modes The output modes for each core.
  /// @param n_modes The input modes for each core.
  /// @param device The device to put this on.
  /// @param dtype The data type for the cores.
  /// @return The resulting TT-matrix.
  static Ptr zeros(const c10::SmallVector<int64_t, 6>& m_modes,
    const c10::SmallVector<int64_t, 6>& n_modes,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt);
  /// @brief Create a TT-matrix of ones.
  /// @param m_modes The output modes for each core.
  /// @param n_modes The input modes for each core.
  /// @param device The device to put this on.
  /// @param dtype The data type for the cores.
  /// @return The resulting TT-matrix.
  static Ptr ones(const c10::SmallVector<int64_t, 6>& m_modes,
    const c10::SmallVector<int64_t, 6>& n_modes,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt);

  /// @brief Perform a transpose (in-place).
  /// @param cores The indices of which cores to index.
  /// @return The transposed TT.
  void transpose_(const c10::SmallVector<int64_t, 6>& core_idxs = {});

  /// @brief Perform a transpose.
  /// @param cores The indices of which cores to index.
  /// @return The transposed TT.
  Ptr transpose(const c10::SmallVector<int64_t, 6>& core_idxs = {}) const;

  // =================================================================
  // Public iterators
  /// @brief Index the vector of TT-cores.
  /// @param i The index position.
  /// @return A constant reference to the core.
  const torch::Tensor& operator[](size_t i) const { return tt_matrix_[i]; }
  /// @brief Index the vector of TT-cores.
  /// @param i The index position.
  /// @return A reference to the core.
  torch::Tensor& operator[](size_t i) { return tt_matrix_[i]; }

  /// @return Beginning iterator through the TT-cores.
  TTEngine::Tensors::const_iterator begin() const { return tt_matrix_.begin(); }
  /// @return Ending iterator through the TT-cores.
  TTEngine::Tensors::const_iterator end() const { return tt_matrix_.end(); }

  // =================================================================
  // Public getters / setters
  /// @return The TT-matrix backend.
  const TTEngine& get_engine() const noexcept { return tt_matrix_; }
  /// @return The number of cores in the TT.
  size_t size() const noexcept { return tt_matrix_.size(); }
  /// @return A vector of the TT-cores.
  const TTEngine::Tensors& get_cores() const noexcept
  {
    return tt_matrix_.get_cores();
  }
  /// @return Get the ranks of the TT-matrix.
  c10::SmallVector<int64_t, 7> get_ranks() const
  {
    return tt_matrix_.get_ranks();
  }
  /// @return Get the sizes of the free indices for the TT-vector.
  c10::SmallVector<int64_t, 12> get_free_indices() const
  {
    return tt_matrix_.get_free_indices();
  }
  /// @brief Get the M mode sizes for this MxN TT-matrix.
  c10::SmallVector<int64_t, 6> get_m_modes() const
  {
    return tt_matrix_.get_m_modes();
  }
  /// @brief Get the N mode sizes for this MxN TT-matrix.
  c10::SmallVector<int64_t, 6> get_n_modes() const
  {
    return tt_matrix_.get_n_modes();
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

// ===================================================================
// Direct sum operators
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTOperator::Ptr>::type
operator+(const T& a, const TTOperator::Ptr& b)
{
  return TTOperator::create(a + b->get_engine());
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTOperator::Ptr>::type
operator+(const TTOperator::Ptr& a, const T& b)
{
  return TTOperator::create(a->get_engine() + b);
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTOperator::Ptr>::type
operator-(const T& a, const TTOperator::Ptr& b)
{
  return TTOperator::create(a - b->get_engine());
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTOperator::Ptr>::type
operator-(const TTOperator::Ptr& a, const T& b)
{
  return TTOperator::create(a->get_engine() - b);
}

inline TTOperator::Ptr operator+(
  const TTOperator::Ptr& a, const TTOperator::Ptr& b)
{
  return TTOperator::create(a->get_engine() + b->get_engine());
}
inline TTOperator::Ptr operator-(
  const TTOperator::Ptr& a, const TTOperator::Ptr& b)
{
  return TTOperator::create(a->get_engine() - b->get_engine());
}

// ===================================================================
// Hadamard (element-wise) multiplication
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTOperator::Ptr>::type
operator*(const T& a, const TTOperator::Ptr& b)
{
  return TTOperator::create(a * b->get_engine());
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTOperator::Ptr>::type
operator*(const TTOperator::Ptr& a, const T& b)
{
  return TTOperator::create(a->get_engine() * b);
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTOperator::Ptr>::type
operator/(const T& a, const TTOperator::Ptr& b)
{
  return TTOperator::create(a / b->get_engine());
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTOperator::Ptr>::type
operator/(const TTOperator::Ptr& a, const T& b)
{
  return TTOperator::create(a->get_engine() / b);
}

inline TTOperator::Ptr operator*(
  const TTOperator::Ptr& a, const TTOperator::Ptr& b)
{
  return TTOperator::create(a->get_engine() * b->get_engine());
}
inline TTOperator::Ptr operator/(
  const TTOperator::Ptr& a, const TTOperator::Ptr& b)
{
  return TTOperator::create(a->get_engine() / b->get_engine());
}

} // namespace ttnte::linalg
