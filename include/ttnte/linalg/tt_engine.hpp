#pragma once

#include <limits>
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
  TTEngine() = default;
  TTEngine(const Tensors& cores, bool check_cores = true);

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

  /// @brief TT-SVD algorithm. Decompose a tensor into a tensor train.
  /// @param tensor The tensor to be approximately decomposed.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum rank to truncate to.
  /// @return The approximate TT representation of the original tensor.
  static Tensors tt_svd(torch::Tensor tensor, double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max());

  /// @brief In-place left-to-right orthogonalization.
  TTEngine& lr_orthogonalize_();
  /// @brief Left-to-right orthogonalization.
  TTEngine lr_orthogonalize() const;

  /// @brief In-place rounding. Reduce the ranks to a maximum of prescribed
  /// error.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum rank to truncate to.
  TTEngine& round_(
    double eps = 1e-12, int64_t max_rank = std::numeric_limits<int64_t>::max());
  /// @brief TT-rounding. Reduce the ranks to a maximum of prescribed
  /// error.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum rank to truncate to.
  TTEngine round(double eps = 1e-12,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const;

  /// @brief Decompose a tensor into a TT-vector.
  /// @param tensor The tensor to be decomposed.
  /// @return The backend for a TT-vector that represents the original tensor
  /// with some error.
  static TTEngine from_dense(const torch::Tensor& tensor, double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max());

  /// @brief Decompose a tensor into a TT-matrix with the given shape.
  /// @param m_modes The output modes for each core.
  /// @param n_modes The input modes for each core.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum rank to truncate to.
  /// @param is_interleaved Whether the tensor is already permuted to
  /// (n_0, m_0, n_1, m_1, ...) or is (n_0, n_1, ..., m_0, m_1, ...)
  /// @return The backend for a TT-matrix that represents the original tensor
  /// with some error.
  static TTEngine from_dense(torch::Tensor tensor,
    const c10::SmallVector<int64_t, 6>& m_modes,
    const c10::SmallVector<int64_t, 6>& n_modes, double eps = 1e-10,
    int64_t max_rank = std::numeric_limits<int64_t>::max(),
    bool is_interleaved = false);

  /// @brief Contract rank dimensions to make a full tensor.
  /// @return The dense tensor.
  torch::Tensor to_dense() const;

  /// @brief Update the TT from a buffer.
  /// @param buffer The buffer of data holding all the TT cores.
  void from_buffer(const torch::Tensor& buffer);

  /// @brief Flip the sign (in-place) of this TT.
  /// @return The reference to this TT with its sign flipped.
  TTEngine& neg_();

  /// @brief Create a TT-vector of zeros.
  /// @param m_modes The modes for each core.
  /// @param device The device to put this on.
  /// @param dtype The data type for the cores.
  /// @return The resulting TT-vector.
  static TTEngine zeros(const c10::SmallVector<int64_t, 6>& m_modes,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt);
  /// @brief Create a TT-matrix of zeros.
  /// @param m_modes The output modes for each core.
  /// @param n_modes The input modes for each core.
  /// @param device The device to put this on.
  /// @param dtype The data type for the cores.
  /// @return The resulting TT-matrix.
  static TTEngine zeros(const c10::SmallVector<int64_t, 6>& m_modes,
    const c10::SmallVector<int64_t, 6>& n_modes,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt);
  /// @brief Create a TT-vector of ones.
  /// @param m_modes The modes for each core.
  /// @param device The device to put this on.
  /// @param dtype The data type for the cores.
  /// @return The resulting TT-vector.
  static TTEngine ones(const c10::SmallVector<int64_t, 6>& m_modes,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt);
  /// @brief Create a TT-matrix of ones.
  /// @param m_modes The output modes for each core.
  /// @param n_modes The input modes for each core.
  /// @param device The device to put this on.
  /// @param dtype The data type for the cores.
  /// @return The resulting TT-matrix.
  static TTEngine ones(const c10::SmallVector<int64_t, 6>& m_modes,
    const c10::SmallVector<int64_t, 6>& n_modes,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt);

  /// @brief Perform a transpose (in-place).
  /// @param cores The indices of which cores to index.
  /// @return The transposed TT.
  TTEngine& transpose_(const c10::SmallVector<int64_t, 6>& core_idxs = {});

  /// @brief Perform a transpose.
  /// @param cores The indices of which cores to index.
  /// @return The transposed TT.
  TTEngine transpose(const c10::SmallVector<int64_t, 6>& core_idxs = {}) const;

  // =================================================================
  // Public operators
  /// @brief Flip the sign of this TT.
  /// @return A new TT with its sign flipped compared to the original.
  TTEngine operator-() const;

  /// @brief In-place scalar addition.
  /// @param The other scalar object.
  /// @return Resulting TT.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    TTEngine&>::type
  operator+=(const T& other)
  {
    // If it's a tensor, ensure it's a single element (scalar)
    if constexpr (std::is_same_v<T, torch::Tensor>) {
      TORCH_CHECK(
        other.numel() == 1, "Only scalar tensors can be added to TTEngine.");
    }
    int64_t num_cores = cores_.size();

    for (size_t i = 0; i < num_cores; i++) {
      auto& core = cores_[i];
      auto shape = core.sizes().vec();

      // Get the current ranks
      int64_t rl = shape[0];
      int64_t rr = shape[3];

      // Update the ranks
      shape[0] += (i == 0) ? 0 : 1;
      shape[3] += (i == num_cores - 1) ? 0 : 1;

      // Create a new core and fill with old
      torch::Tensor new_core = torch::zeros(shape, core.options());
      new_core.slice(0, 0, rl).slice(3, 0, rr) = core;

      // Get the updated indices to put the value in the bottom right corner of
      // the core
      int64_t row_idx = (i == 0) ? 0 : rl;
      int64_t col_idx = (i == num_cores - 1) ? 0 : rr;
      if (i == 0) {
        new_core.select(0, row_idx).select(2, col_idx).fill_(other);
      } else {
        new_core.select(0, row_idx).select(2, col_idx).fill_(1.0);
      }
      core = new_core;
    }

    return *this;
  }
  /// @brief In-place scalar subtraction.
  /// @param The other scalar object.
  /// @return Resulting TT.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    TTEngine&>::type
  operator-=(const T& other)
  {
    return *this += (-other);
  }
  /// @brief In-place scalar multiplication.
  /// @param The other scalar object.
  /// @return Resulting TT.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    TTEngine&>::type
  operator*=(const T& other)
  {
    cores_[0] = cores_[0] * other;
    return *this;
  }
  /// @brief In-place scalar division.
  /// @param The other scalar object.
  /// @return Resulting TT.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    TTEngine&>::type
  operator/=(const T& other)
  {
    cores_[0] = cores_[0] / other;
    return *this;
  }

  /// @brief In-place TT addition.
  /// @param The other TT object.
  /// @return Resulting TT.
  TTEngine& operator+=(const TTEngine& other);
  /// @brief In-place TT subtraction.
  /// @param The other TT object.
  /// @return Resulting TT.
  TTEngine& operator-=(const TTEngine& other);
  /// @brief In-place TT Hadamard multiplication.
  /// @param The other TT object.
  /// @return Resulting TT.
  TTEngine& operator*=(const TTEngine& other);
  /// @brief In-place TT Hadamard division. Note this is approximate and uses
  /// AMEn. The arguments for the AMEn solver is fixed so use the
  /// `elementwise_divide`
  /// @param The other TT object.
  /// @return Resulting TT.
  TTEngine& operator/=(const TTEngine& other);

  // =================================================================
  // Public iterators
  /// @brief Index the vector of TT-cores.
  /// @param i The index position.
  /// @return A constant reference to the core.
  const torch::Tensor& operator[](size_t i) const { return cores_[i]; }
  /// @brief Index the vector of TT-cores.
  /// @param i The index position.
  /// @return A reference to the core.
  torch::Tensor& operator[](size_t i) { return cores_[i]; }

  /// @return Beginning iterator through the TT-cores.
  Tensors::const_iterator begin() const { return cores_.begin(); }
  /// @return Ending iterator through the TT-cores.
  Tensors::const_iterator end() const { return cores_.end(); }

  // =================================================================
  // Public getters / setters
  /// @return The number of cores in the TT.
  size_t size() const noexcept { return cores_.size(); }
  /// @return Get the TT-cores.
  const Tensors& get_cores() const noexcept { return cores_; }
  /// @return Get the device of the TT.
  torch::Device get_device() const noexcept { return cores_[0].device(); }
  /// @return Get the data type of the TT.
  at::ScalarType get_dtype() const noexcept { return cores_[0].scalar_type(); }
  /// @return Get the storage size of the TT.
  int64_t get_numel() const;
  /// @return Get the TT-ranks.
  c10::SmallVector<int64_t, 7> get_ranks() const;
  /// @return Get the size of the free indices.
  c10::SmallVector<int64_t, 12> get_free_indices() const;
  /// @return The size of the output dimensions for each core shaped like MxN.
  c10::SmallVector<int64_t, 6> get_m_modes() const;
  /// @return The size of the input dimensions for each core shaped like MxN.
  c10::SmallVector<int64_t, 6> get_n_modes() const;
};

// ===================================================================
// Direct sum operators
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator+(const T& a, TTEngine b)
{
  b += a;
  return b;
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator+(TTEngine a, const T& b)
{
  a += b;
  return a;
}

template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator-(const T& a, const TTEngine& b)
{
  return a + (-b);
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator-(TTEngine a, const T& b)
{
  return a -= b;
}

inline TTEngine operator+(TTEngine a, const TTEngine& b)
{
  return a += b;
}
inline TTEngine operator-(TTEngine a, const TTEngine& b)
{
  return a -= b;
}

// ===================================================================
// Hadamard (element-wise) multiplication
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator*(const T& a, TTEngine b)
{
  b *= a;
  return b;
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator*(TTEngine a, const T& b)
{
  a *= b;
  return a;
}
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator/(const T& a, const TTEngine& b);

template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator/(TTEngine a, const T& b)
{
  a /= b;
  return a;
}

inline TTEngine operator*(TTEngine a, const TTEngine& b)
{
  return a *= b;
}
TTEngine operator/(const TTEngine& a, const TTEngine& b);

} // namespace ttnte::linalg
