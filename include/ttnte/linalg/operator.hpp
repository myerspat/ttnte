#pragma once

#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/label.hpp"
#include <ATen/core/ivalue.h>
#include <optional>
#include <string>
#include <variant>

/// @file
/// @brief Defines the Operator and OperatorData classes for representing
/// mathematical operators in tensor-train format and others.

namespace ttnte::linalg {

/// @class OperatorData
/// @brief Underlying data storage for an operator, compatible with libtorch's
/// custom class system.
///
/// This class inherits from `torch::CustomClassHolder` to allow it to be safely
/// managed by libtorch's intrusive pointer system (c10::intrusive_ptr). It
/// wraps a variant containing the actual tensor engine representing a
/// matrix/operator.
class OperatorData : public torch::CustomClassHolder {
public:
  // =================================================================
  // Public types
  /// @brief Variant type used to store different underlying engine types.
  using MatrixVariant = std::variant<TTEngine>;

  // =================================================================
  // Public data
  /// Storage for the underlying matrix/tensor engine.
  MatrixVariant matrix_;

  // =================================================================
  // Public constructors
  /// @brief Construct a new OperatorData object.
  /// @param matrix The underlying matrix variant to store.
  OperatorData(MatrixVariant matrix) : matrix_(std::move(matrix)) {}
};

/// @class Operator
/// @brief High-level handle for a physical or mathematical operator (matrix).
///
/// The Operator class provides a safe, reference-counted interface to the
/// underlying `OperatorData`. It supports a wide array of mathematical
/// operations, device transfers (CPU/GPU), and datatype casting similar to
/// `torch::Tensor`.
class Operator {
public:
  // =================================================================
  // Public types
  /// @brief Alias for the label type used to identify this operator.
  using Label = utils::Label<Operator>;

private:
  // =================================================================
  // Private data
  /// Label of the operator for debugging and tracking.
  Label label_;
  /// Data storage managed via libtorch's intrusive pointer.
  c10::intrusive_ptr<OperatorData> data_;

  // =================================================================
  // Private constructors
  /// @brief Internal constructor for creating an Operator from existing data
  /// and a label.
  /// @param data Intrusive pointer to the allocated OperatorData.
  /// @param label The label to assign to this Operator.
  explicit Operator(c10::intrusive_ptr<OperatorData> data, Label label)
    : data_(std::move(data)), label_(std::move(label))
  {}

  // =================================================================
  // Private methods
  /// @brief Generates an error context string for exception throwing.
  /// @param func_name The name of the function where the error occurred.
  /// @return Formatted string representing the function context.
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::linalg::Operator::" + func_name;
  }

public:
  // =================================================================
  // Public constructors
  /// @brief Default constructor creating an undefined Operator.
  Operator() noexcept : data_(nullptr) {}

  /// @brief Default copy constructor (shallow copy of the intrusive pointer).
  Operator(const Operator& other) = default;

  /// @brief Default copy assignment operator.
  Operator& operator=(const Operator& other) = default;

  /// @brief Construct a new Operator from a Tensor Train engine.
  /// @param engine The TTEngine representing a TT-matrix (operator).
  /// @param label Optional string label for the operator. If omitted, an
  /// internal label is generated.
  Operator(TTEngine engine, std::optional<std::string> label = std::nullopt)
    : data_(c10::make_intrusive<OperatorData>(std::move(engine))),
      label_(label.has_value() ? Label::from_string(*label)
                               : Label::create_internal())
  {}

  // =================================================================
  // Public methods
  /// @brief Check if the operator holds valid data.
  /// @return True if the internal data pointer is not null, false otherwise.
  [[nodiscard]] bool defined() const noexcept { return data_ != nullptr; }

  /// @brief Serialize the operator data into a contiguous torch::Tensor buffer.
  /// @param buffer The destination tensor buffer.
  void to_buffer(const torch::Tensor& buffer) const;

  /// @brief Deserialize the operator data from a contiguous torch::Tensor
  /// buffer.
  /// @param buffer The source tensor buffer.
  void from_buffer(const torch::Tensor& buffer);

  /// @brief In-place conversion of the operator's tensor options.
  /// @param options Target torch::TensorOptions.
  /// @param non_blocking If true, tries to perform the transfer asynchronously.
  /// @param copy If true, forces a copy even if the options match.
  /// @return Reference to the modified Operator.
  Operator& to_(const torch::TensorOptions& options, bool non_blocking = false,
    bool copy = false);

  /// @brief In-place conversion specifying device and dtype.
  Operator& to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief In-place conversion specifying dtype only.
  Operator& to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief In-place conversion specifying device only.
  Operator& to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief Out-of-place conversion of the operator's tensor options.
  /// @param options Target torch::TensorOptions.
  /// @param non_blocking If true, tries to perform the transfer asynchronously.
  /// @param copy If true, forces a copy even if the options match.
  /// @return A new Operator with the applied options.
  Operator to(const torch::TensorOptions& options, bool non_blocking = false,
    bool copy = false) const;

  /// @brief Out-of-place conversion specifying device and dtype.
  Operator to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Out-of-place conversion specifying dtype only.
  Operator to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Out-of-place conversion specifying device only.
  Operator to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief In-place negation of the operator.
  /// @return Reference to the modified Operator.
  Operator& neg_();

  /// @brief In-place rounding/truncation of the Tensor Train ranks.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum allowed rank after rounding.
  /// @return Reference to the modified Operator.
  Operator& round_(
    double eps = 1e-12, int64_t max_rank = std::numeric_limits<int64_t>::max());

  /// @brief Out-of-place rounding/truncation of the Tensor Train ranks.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum allowed rank after rounding.
  /// @return A new Operator with rounded ranks.
  Operator round(double eps = 1e-12,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const;

  /// @brief In-place restriction of one mode to a sub-range of length `length`.
  /// @param dim  With interleaved=false: dim < K narrows the m-mode of core
  /// dim;
  ///             dim >= K narrows the n-mode of core (dim-K).
  ///             With interleaved=true: even dim narrows m-mode, odd narrows
  ///             n-mode. Negative start values are resolved modulo the mode
  ///             size.
  /// @param start  Start index (negative counts from end).
  /// @param length  Length of the slice (default 1).
  /// @param interleaved  Dimension convention (default false = uninterleaved).
  /// @return Reference to the modified Operator.
  Operator& narrow_(
    size_t dim, int64_t start, int64_t length = 1, bool interleaved = false);

  /// @brief Out-of-place version of narrow_.
  /// @param dim  See narrow_ documentation.
  /// @param start  Start index (negative counts from end).
  /// @param length  Length of the slice (default 1).
  /// @param interleaved  Dimension convention (default false = uninterleaved).
  /// @return A new Operator with the mode narrowed.
  Operator narrow(size_t dim, int64_t start, int64_t length = 1,
    bool interleaved = false) const;

  /// @brief Reverse the index ordering of one or more modes in-place.
  /// Follows the same dim/interleaved convention as narrow_().
  /// Multiple dims mapping to the same core are grouped into one flip call.
  /// @param dims        Axis indices in the chosen layout.
  /// @param interleaved Whether dims follow the interleaved layout.
  /// @return Reference to the modified Operator.
  Operator& flip_(at::IntArrayRef dims, bool interleaved = false);

  /// @brief Out-of-place version of flip_().
  /// @param dims        Axis indices in the chosen layout.
  /// @param interleaved Whether dims follow the interleaved layout.
  /// @return A new Operator with the specified modes flipped.
  Operator flip(at::IntArrayRef dims, bool interleaved = false) const;

  // =================================================================
  // Public operators

  /// @brief Unary negation operator.
  Operator operator-() const
  {
    auto new_variant = std::visit(
      [&](const auto& v) -> OperatorData::MatrixVariant { return -v; },
      get_variant());

    return Operator(c10::make_intrusive<OperatorData>(std::move(new_variant)),
      label_.is_user_defined() ? label_.clone() : Label::create_internal());
  }

  /// @brief In-place addition with a scalar or tensor.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    Operator&>::type
  operator+=(const T& other)
  {
    std::visit([&](auto& v) { v += other; }, get_variant());
    return *this;
  }

  /// @brief In-place subtraction with a scalar or tensor.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    Operator&>::type
  operator-=(const T& other)
  {
    *this += (-other);
    return *this;
  }

  /// @brief In-place multiplication with a scalar or tensor.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    Operator&>::type
  operator*=(const T& other)
  {
    std::visit([&](auto& v) { v *= other; }, get_variant());
    return *this;
  }

  /// @brief In-place division by a scalar or tensor.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    Operator&>::type
  operator/=(const T& other)
  {
    std::visit([&](auto& v) { v /= other; }, get_variant());
    return *this;
  }

  /// @brief In-place addition with another Operator.
  Operator& operator+=(const Operator& other)
  {
    std::visit([&](auto& v0, const auto& v1) { v0 += v1; }, get_variant(),
      other.get_variant());
    return *this;
  }

  /// @brief In-place subtraction with another Operator.
  Operator& operator-=(const Operator& other)
  {
    std::visit([&](auto& v0, const auto& v1) { v0 -= v1; }, get_variant(),
      other.get_variant());
    return *this;
  }

  /// @brief In-place multiplication with another Operator.
  Operator& operator*=(const Operator& other)
  {
    std::visit([&](auto& v0, const auto& v1) { v0 *= v1; }, get_variant(),
      other.get_variant());
    return *this;
  }

  /// @brief In-place division by another Operator.
  Operator& operator/=(const Operator& other)
  {
    std::visit([&](auto& v0, const auto& v1) { v0 /= v1; }, get_variant(),
      other.get_variant());
    return *this;
  }

  // =================================================================
  // Public getters / setters

  /// @brief Get the label of the operator.
  const Label& get_label() const noexcept { return label_; }

  /// @brief Get the underlying OperatorData (const).
  /// @throws ttnte::utils::runtime_error If the Operator is undefined.
  const c10::intrusive_ptr<OperatorData>& get_data() const
  {
    if (defined()) {
      return data_;
    }

    throw utils::runtime_error(*this, error_context("get_data"),
      "Attempted to access data of an undefined Operator");
  }

  /// @brief Get the underlying OperatorData (mutable).
  /// @throws ttnte::utils::runtime_error If the Operator is undefined.
  c10::intrusive_ptr<OperatorData>& get_data()
  {
    if (defined()) {
      return data_;
    }

    throw utils::runtime_error(*this, error_context("get_data"),
      "Attempted to access data of an undefined Operator");
  }

  /// @brief Get the variant storing the underlying engine (const).
  const OperatorData::MatrixVariant& get_variant() const
  {
    return get_data()->matrix_;
  }

  /// @brief Get the variant storing the underlying engine (mutable).
  OperatorData::MatrixVariant& get_variant() { return get_data()->matrix_; }

  /// @brief Set the label for this operator.
  void set_label(const Label& label) { label_ = label; }

  /// @brief Check if the underlying engine is on a CUDA device.
  bool is_cuda() const { return get_device().is_cuda(); }

  /// @brief Get the PyTorch device where the engine's data resides.
  torch::Device get_device() const
  {
    return std::visit(
      [](const auto& v) { return v.get_device(); }, get_variant());
  }

  /// @brief Get the PyTorch scalar type of the engine's data.
  at::ScalarType get_dtype() const
  {
    return std::visit(
      [](const auto& v) { return v.get_dtype(); }, get_variant());
  }

  /// @brief Get the total number of elements represented by the operator.
  int64_t get_numel() const
  {
    return std::visit(
      [](const auto& v) { return v.get_numel(); }, get_variant());
  }

  /// @brief Check if the operator is represented by a Tensor Train engine.
  bool is_tt() const { return std::holds_alternative<TTEngine>(get_variant()); }
  /// @return The number of logical dimensions: m-modes + n-modes.
  /// For TT operators: 2 * K (both m and n modes are physical).
  /// For dense operators: the tensor's ndimension().
  int64_t ndimension() const;

  /// @brief Retrieve the underlying TTEngine.
  /// @throws ttnte::utils::runtime_error If the operator does not contain a
  /// TTEngine.
  const TTEngine& as_tt() const
  {
    if (auto* tt = std::get_if<TTEngine>(&get_variant()))
      return *tt;
    throw utils::runtime_error(
      *this, error_context("as_tt"), "Operator does not hold a TTEngine");
  }
};

// =================================================================
// Free Functions / Operators for Operators
// =================================================================

/// @brief Addition of an Operator and a scalar/tensor.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  Operator>::type
operator+(const Operator& a, const T& b)
{
  return std::visit([&](const auto& v) -> Operator { return Operator(v + b); },
    a.get_variant());
}

/// @brief Addition of a scalar/tensor and an Operator.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  Operator>::type
operator+(const T& a, const Operator& b)
{
  return b + a;
}

/// @brief Addition of two Operators.
inline Operator operator+(const Operator& a, const Operator& b)
{
  return std::visit([](const auto& v0,
                      const auto& v1) -> Operator { return Operator(v0 + v1); },
    a.get_variant(), b.get_variant());
}

/// @brief Subtraction of a scalar/tensor from an Operator.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  Operator>::type
operator-(const Operator& a, const T& b)
{
  return a + (-b);
}

/// @brief Subtraction of an Operator from a scalar/tensor.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  Operator>::type
operator-(const T& a, const Operator& b)
{
  return std::visit([&](const auto& v) -> Operator { return Operator(a - v); },
    b.get_variant());
}

/// @brief Subtraction of two Operators.
inline Operator operator-(const Operator& a, const Operator& b)
{
  return std::visit([&](const auto& v0,
                      const auto& v1) -> Operator { return Operator(v0 - v1); },
    a.get_variant(), b.get_variant());
}

/// @brief Multiplication of an Operator and a scalar/tensor.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  Operator>::type
operator*(const Operator& a, const T& b)
{
  return std::visit([&](const auto& v) -> Operator { return Operator(v * b); },
    a.get_variant());
}

/// @brief Multiplication of a scalar/tensor and an Operator.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  Operator>::type
operator*(const T& a, const Operator& b)
{
  return b * a;
}

/// @brief Multiplication of two Operators.
inline Operator operator*(const Operator& a, const Operator& b)
{
  return std::visit([](const auto& v0,
                      const auto& v1) -> Operator { return Operator(v0 * v1); },
    a.get_variant(), b.get_variant());
}

/// @brief Division of an Operator by a scalar/tensor.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  Operator>::type
operator/(const Operator& a, const T& b)
{
  return std::visit([&](const auto& v) -> Operator { return Operator(v / b); },
    a.get_variant());
}

/// @brief Division of a scalar/tensor by an Operator.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  Operator>::type
operator/(const T& a, const Operator& b)
{
  return std::visit([&](const auto& v) -> Operator { return Operator(a / v); },
    b.get_variant());
}

/// @brief Division of two Operators.
inline Operator operator/(const Operator& a, const Operator& b)
{
  return std::visit([](const auto& v0,
                      const auto& v1) -> Operator { return Operator(v0 / v1); },
    a.get_variant(), b.get_variant());
}

} // namespace ttnte::linalg
