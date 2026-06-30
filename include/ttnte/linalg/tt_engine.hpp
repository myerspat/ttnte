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
  /// type (in-place).
  /// @param options Tensor options to apply to each TT-core.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  TTEngine to(const torch::TensorOptions& options, bool non_blocking = false,
    bool copy = false) const;
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
  /// @param options Tensor options to apply to each TT-core.
  /// @param non_blocking Whether to run this method as non-blocking.
  /// @param copy Whether to copy the TT.
  TTEngine& to_(const torch::TensorOptions& options, bool non_blocking = false,
    bool copy = false);
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

  /// @return The sum over all indices.
  double sum() const;
  /// @return The Frobenius norm of this TT.
  double norm() const;

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
  /// @param interleave Whether to leave the tensor interleaved or group indices
  /// by (output, input).
  /// @return The dense tensor.
  torch::Tensor to_dense(bool interleave = false) const;

  /// @brief Add the info for the TT to the buffer.
  /// @param buffer The buffer to append the information.
  void to_buffer(const torch::Tensor& buffer) const;

  /// @brief Update the TT from a buffer.
  /// @param buffer The buffer of data holding all the TT cores.
  void from_buffer(const torch::Tensor& buffer);

  /// @brief Serialize the TT to a flat CPU tensor suitable for MPI.
  /// Wire format: [FormatType::TENSOR_TRAIN, K, core0_shape[4], ..., core
  /// data].
  /// @param buffer If defined, pack into this pre-allocated 1D CPU tensor (its
  ///   dtype is used; core data is cast accordingly). If undefined (default),
  ///   a new buffer is allocated in the TT's native dtype.
  /// @return The buffer containing the serialized TT.
  torch::Tensor pack(const torch::Tensor& buffer = torch::Tensor()) const;

  /// @brief Deserialize a TTEngine from a flat CPU tensor produced by pack().
  /// The reconstructed TT has the same dtype as the buffer.
  /// @param buffer A 1D CPU tensor produced by pack().
  /// @return A new TTEngine in the dtype encoded in the buffer.
  static TTEngine unpack(const torch::Tensor& buffer);

  /// @brief Flip the sign (in-place) of this TT.
  /// @return The reference to this TT with its sign flipped.
  TTEngine& neg_();

  /// @brief Restrict a dimension to a contiguous sub-range in-place.
  /// `dim` is interpreted relative to the layout produced by
  /// `to_dense(interleaved)`:
  ///   - `interleaved=false` (default): layout [m_0,…,m_{K-1},n_0,…,n_{K-1}].
  ///     dim < K narrows the m-mode of core dim; dim >= K narrows the n-mode
  ///     of core dim-K.
  ///   - `interleaved=true`: layout [m_0,n_0,m_1,n_1,…].
  ///     Even dim narrows the m-mode of core dim/2; odd dim narrows the
  ///     n-mode of core dim/2.
  /// `start` may be negative to index from the end of the mode.
  /// @param dim    Axis index in the chosen layout.
  /// @param start  First index to keep; negative counts from the end.
  /// @param length Number of indices to keep.
  /// @param interleaved Whether dim follows the interleaved layout.
  /// @return Reference to this TT after narrowing.
  TTEngine& narrow_(
    size_t dim, int64_t start, int64_t length = 1, bool interleaved = false);

  /// @brief Restrict a dimension to a contiguous sub-range (out-of-place).
  /// @param dim    Axis index in the chosen layout.
  /// @param start  First index to keep; negative counts from the end.
  /// @param length Number of indices to keep.
  /// @param interleaved Whether dim follows the interleaved layout.
  /// @return A new TTEngine with the specified dimension narrowed.
  TTEngine narrow(size_t dim, int64_t start, int64_t length = 1,
    bool interleaved = false) const;

  /// @brief Reverse the index ordering of one or more modes in-place.
  /// `dims` follows the same layout convention as narrow_():
  ///   - `interleaved=false` (default): dim < K flips the m-mode of core dim;
  ///     dim >= K flips the n-mode of core dim-K.
  ///   - `interleaved=true`: even dim flips the m-mode of core dim/2;
  ///     odd dim flips the n-mode of core dim/2.
  /// Multiple dims mapping to the same core are grouped into one flip call.
  /// @param dims        Axis indices in the chosen layout.
  /// @param interleaved Whether dims follow the interleaved layout.
  /// @return Reference to this TT after flipping.
  TTEngine& flip_(at::IntArrayRef dims, bool interleaved = false);

  /// @brief Reverse the index ordering of one or more modes (out-of-place).
  /// @param dims        Axis indices in the chosen layout.
  /// @param interleaved Whether dims follow the interleaved layout.
  /// @return A new TTEngine with the specified modes flipped.
  TTEngine flip(at::IntArrayRef dims, bool interleaved = false) const;

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

  /// @brief Permute the ordering of TT cores in-place.
  /// @param dims      New order of core indices; dims[k] = source core at
  /// position k.
  /// @param eps       SVD truncation tolerance for each adjacent swap (default
  /// 0 = lossless).
  /// @param max_rank  Maximum bond dimension after each swap.
  /// @return Reference to this TT after permuting.
  TTEngine& permute_(at::IntArrayRef dims, double eps = 0.0,
    int64_t max_rank = std::numeric_limits<int64_t>::max());

  /// @brief Permute the ordering of TT cores (out-of-place).
  /// @param dims      New order of core indices; dims[k] = source core at
  /// position k.
  /// @param eps       SVD truncation tolerance for each adjacent swap (default
  /// 0 = lossless).
  /// @param max_rank  Maximum bond dimension after each swap.
  /// @return A new TT with cores in the requested order.
  TTEngine permute(at::IntArrayRef dims, double eps = 0.0,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const;

  /// @brief Diagonalize a select set of cores (in-place). This assumes the
  /// cores are already for a TT-vector.
  /// @param core_idxs Indices into the cores vector of which we are to
  /// diagonalize.
  TTEngine& diagonalize_(const c10::SmallVector<int64_t, 6>& core_idxs = {});
  /// @brief Diagonalize a select set of cores. This assumes the cores are
  /// already for a TT-vector.
  /// @param core_idxs Indices into the cores vector of which we are to
  /// diagonalize.
  TTEngine diagonalize(
    const c10::SmallVector<int64_t, 6>& core_idxs = {}) const;

  /// @return Whether the TT is rank-one.
  bool is_rank_one() const;
  /// @return Whether the TT is a TT-vector.
  bool is_vector() const;

  /// @brief Expand a core along a given dimension (in-place). The values across
  /// the other dimensions are repeated for the new one.
  /// @param m_modes The shape of the output dimensions of the TT.
  /// @param n_modes The shape of the input dimensions of the TT.
  /// @return A reference to this TT after the expansion.
  TTEngine& expand_(const c10::SmallVector<int64_t, 6>& m_modes,
    const c10::SmallVector<int64_t, 6>& n_modes);
  /// @brief Expand a core along a given dimension. The values across
  /// the other dimensions are repeated for the new one.
  /// @param m_modes The shape of the output dimensions of the TT.
  /// @param n_modes The shape of the input dimensions of the TT.
  /// @return The expanded TT.
  TTEngine expand(const c10::SmallVector<int64_t, 6>& m_modes,
    const c10::SmallVector<int64_t, 6>& n_modes) const;

  /// @brief Contract a specific rank dimension (in-place).
  /// @param dim The index of which dimension to contract (not including the end
  /// ranks).
  /// @return A reference to this TT after the rank dimension has been
  /// contracted.
  TTEngine& contract_rank_dim_(size_t dim);
  /// @brief Contract a specific rank dimension.
  /// @param dim The index of which dimension to contract (not including the end
  /// ranks).
  /// @return The TT after the rank dimension has been contracted.
  TTEngine contract_rank_dim(size_t dim) const;

  /// @brief Create a mesh grid (like numpy or pytorch) but in TT format.
  /// @param vecs The grid along each dimension.
  /// @return Each tensor in vecs seperated into their own TT with ones in the
  /// other dimensions.
  static c10::SmallVector<linalg::TTEngine, 6> meshgrid(const Tensors& vecs);

  /// @brief Compute the Kronecker product of thie TT with another TT
  /// (in-place). Note this just combines the two TT core vectors into one.
  /// @param other The other TT.
  /// @return A reference to this TT after the Kronecker product.
  TTEngine& kron_(const TTEngine& other);
  /// @brief Compute the Kronecker product of thie TT with another TT. Note this
  /// just combines the two TT core vectors into one.
  /// @param other The other TT.
  /// @return The resulting TT.
  TTEngine kron(const TTEngine& other) const;
  /// @brief Compute the Kronecker product with another tensor (in-place). This
  /// adds the tensor to the end of the list of TT-cores.
  /// @param other The tensor.
  /// @return A reference to this TT after the Kronecker product.
  TTEngine& kron_(const torch::Tensor& other);
  /// @brief Compute the Kronecker product with another tensor. This
  /// adds the tensor to the end of the list of TT-cores.
  /// @param other The tensor.
  /// @return The resulting TT.
  TTEngine kron(const torch::Tensor& other) const;

  /// @brief Evaluate this TT at a list of indices.
  /// @param indices A tensor that is (N, C) where C is the number of indices in
  /// this TT.
  /// @return The resulting N evaluations.
  torch::Tensor evaluate_at(const torch::Tensor& indices) const;

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
  /// @return The total number of logical modes: 2 * size(), since every core
  /// is 4-D [r_l, m, n, r_r] regardless of whether it represents a state or
  /// operator. State and Operator correct for trivial (size-1) modes in their
  /// own ndimension() implementations.
  int64_t ndimension() const noexcept
  {
    return 2 * static_cast<int64_t>(cores_.size());
  }
  /// @return Get the TT-cores.
  const Tensors& get_cores() const noexcept { return cores_; }
  /// @return Get the device of the TT.
  torch::Device get_device() const noexcept { return cores_[0].device(); }
  /// @return Get the data type of the TT.
  at::ScalarType get_dtype() const noexcept { return cores_[0].scalar_type(); }
  /// @return Get the storage size of the TT.
  int64_t get_numel() const;
  /// @return The compression ratio.
  double get_compression() const;
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
