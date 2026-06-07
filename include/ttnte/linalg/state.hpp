#pragma once

#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/label.hpp"
#include <ATen/core/ivalue.h>
#include <optional>
#include <string>
#include <variant>

/// @file
/// @brief Defines the State and StateData classes for representing mathematical
/// states in tensor-train format.

namespace ttnte::linalg {

/// @class StateData
/// @brief Underlying data storage for a state, compatible with libtorch's
/// custom class system.
///
/// This class inherits from `torch::CustomClassHolder` to allow it to be safely
/// managed by libtorch's intrusive pointer system (c10::intrusive_ptr). It
/// wraps a variant containing the actual tensor engine (e.g., TTEngine).
class StateData : public torch::CustomClassHolder {
public:
  // =================================================================
  // Public types
  /// @brief Variant type used to store different underlying engine types.
  using VectorVariant = std::variant<TTEngine>;

  // =================================================================
  // Public data
  /// Storage for the underlying vector/tensor engine.
  VectorVariant vector_;

  // =================================================================
  // Public constructors
  /// @brief Construct a new StateData object.
  /// @param vector The underlying vector variant to store.
  StateData(VectorVariant vector) : vector_(std::move(vector)) {}
};

/// @class State
/// @brief High-level handle for a physical or mathematical state vector.
///
/// The State class provides a safe, reference-counted interface to the
/// underlying `StateData`. It supports a wide array of mathematical operations,
/// device transfers (CPU/GPU), and datatype casting similar to `torch::Tensor`.
class State {
public:
  // =================================================================
  // Public types
  /// @brief Alias for the label type used to identify this state.
  using Label = utils::Label<State>;

private:
  // =================================================================
  // Private data
  /// Label of the state for debugging and tracking.
  Label label_;
  /// Data storage managed via libtorch's intrusive pointer.
  c10::intrusive_ptr<StateData> data_;

  // =================================================================
  // Private constructors
  /// @brief Internal constructor for creating a State from existing data and a
  /// label.
  /// @param data Intrusive pointer to the allocated StateData.
  /// @param label The label to assign to this State.
  explicit State(c10::intrusive_ptr<StateData> data, Label label)
    : data_(std::move(data)), label_(std::move(label))
  {}

  // =================================================================
  // Private methods
  /// @brief Generates an error context string for exception throwing.
  /// @param func_name The name of the function where the error occurred.
  /// @return Formatted string representing the function context.
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::linalg::State::" + func_name;
  }

public:
  // =================================================================
  // Public constructors
  /// @brief Default constructor creating an undefined State.
  State() noexcept : data_(nullptr) {}

  /// @brief Default copy constructor (shallow copy of the intrusive pointer).
  State(const State& other) = default;

  /// @brief Default copy assignment operator.
  State& operator=(const State& other) = default;

  /// @brief Construct a new State from a Tensor Train engine.
  /// @param engine The TTEngine representing a TT-vector.
  /// @param label Optional string label for the state. If omitted, an internal
  /// label is generated.
  /// @throws ttnte::utils::runtime_error If the given engine does not represent
  /// a valid TT-vector.
  State(TTEngine engine, std::optional<std::string> label = std::nullopt)
    : label_(label.has_value() ? Label::from_string(*label)
                               : Label::create_internal())
  {
    if (!engine.is_vector()) {
      throw utils::runtime_error(error_context("State"),
        "The TT-cores given do not make a "
        "TT-vector, all n-modes should be size 1");
    }
    data_ = c10::make_intrusive<StateData>(std::move(engine));
  }

  // =================================================================
  // Public methods
  /// @brief Check if the state holds valid data.
  /// @return True if the internal data pointer is not null, false otherwise.
  [[nodiscard]] bool defined() const noexcept { return data_ != nullptr; }

  /// @brief Serialize the state data into a contiguous torch::Tensor buffer.
  /// @param buffer The destination tensor buffer.
  void to_buffer(const torch::Tensor& buffer) const;

  /// @brief Deserialize the state data from a contiguous torch::Tensor buffer.
  /// @param buffer The source tensor buffer.
  void from_buffer(const torch::Tensor& buffer);

  /// @brief In-place conversion of the state's tensor options.
  /// @param options Target torch::TensorOptions.
  /// @param non_blocking If true, tries to perform the transfer asynchronously.
  /// @param copy If true, forces a copy even if the options match.
  /// @return Reference to the modified State.
  State& to_(const torch::TensorOptions& options, bool non_blocking = false,
    bool copy = false);

  /// @brief In-place conversion specifying device and dtype.
  State& to_(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief In-place conversion specifying dtype only.
  State& to_(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief In-place conversion specifying device only.
  State& to_(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt);

  /// @brief Out-of-place conversion of the state's tensor options.
  /// @param options Target torch::TensorOptions.
  /// @param non_blocking If true, tries to perform the transfer asynchronously.
  /// @param copy If true, forces a copy even if the options match.
  /// @return A new State with the applied options.
  State to(const torch::TensorOptions& options, bool non_blocking = false,
    bool copy = false) const;

  /// @brief Out-of-place conversion specifying device and dtype.
  State to(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking = false, bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Out-of-place conversion specifying dtype only.
  State to(const at::ScalarType& dtype, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief Out-of-place conversion specifying device only.
  State to(const torch::Device& device, bool non_blocking = false,
    bool copy = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

  /// @brief In-place negation of the state.
  /// @return Reference to the modified State.
  State& neg_();

  /// @brief In-place rounding/truncation of the Tensor Train ranks.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum allowed rank after rounding.
  /// @return Reference to the modified State.
  State& round_(
    double eps = 1e-12, int64_t max_rank = std::numeric_limits<int64_t>::max());

  /// @brief Out-of-place rounding/truncation of the Tensor Train ranks.
  /// @param eps The truncation tolerance.
  /// @param max_rank The maximum allowed rank after rounding.
  /// @return A new State with rounded ranks.
  State round(double eps = 1e-12,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) const;

  // =================================================================
  // Public operators

  /// @brief Unary negation operator.
  State operator-() const
  {
    auto new_variant =
      std::visit([&](const auto& v) -> StateData::VectorVariant { return -v; },
        get_variant());

    return State(c10::make_intrusive<StateData>(std::move(new_variant)),
      label_.is_user_defined() ? label_.clone() : Label::create_internal());
  }

  /// @brief In-place addition with a scalar or tensor.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    State&>::type
  operator+=(const T& other)
  {
    std::visit([&](auto& v) { v += other; }, get_variant());
    return *this;
  }

  /// @brief In-place subtraction with a scalar or tensor.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    State&>::type
  operator-=(const T& other)
  {
    *this += (-other);
    return *this;
  }

  /// @brief In-place multiplication with a scalar or tensor.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    State&>::type
  operator*=(const T& other)
  {
    std::visit([&](auto& v) { v *= other; }, get_variant());
    return *this;
  }

  /// @brief In-place division by a scalar or tensor.
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_same<T, torch::Tensor>::value,
    State&>::type
  operator/=(const T& other)
  {
    std::visit([&](auto& v) { v /= other; }, get_variant());
    return *this;
  }

  /// @brief In-place addition with another State.
  State& operator+=(const State& other)
  {
    std::visit([&](auto& v0, const auto& v1) { v0 += v1; }, get_variant(),
      other.get_variant());
    return *this;
  }

  /// @brief In-place subtraction with another State.
  State& operator-=(const State& other)
  {
    std::visit([&](auto& v0, const auto& v1) { v0 -= v1; }, get_variant(),
      other.get_variant());
    return *this;
  }

  /// @brief In-place multiplication with another State.
  State& operator*=(const State& other)
  {
    std::visit([&](auto& v0, const auto& v1) { v0 *= v1; }, get_variant(),
      other.get_variant());
    return *this;
  }

  /// @brief In-place division by another State.
  State& operator/=(const State& other)
  {
    std::visit([&](auto& v0, const auto& v1) { v0 /= v1; }, get_variant(),
      other.get_variant());
    return *this;
  }

  // =================================================================
  // Public getters / setters

  /// @brief Get the label of the state.
  const Label& get_label() const noexcept { return label_; }

  /// @brief Get the underlying StateData (const).
  /// @throws ttnte::utils::runtime_error If the State is undefined.
  const c10::intrusive_ptr<StateData>& get_data() const
  {
    if (defined()) {
      return data_;
    }
    throw utils::runtime_error(*this, error_context("get_data"),
      "Attempted to access data of an undefined State");
  }

  /// @brief Get the underlying StateData (mutable).
  /// @throws ttnte::utils::runtime_error If the State is undefined.
  c10::intrusive_ptr<StateData>& get_data()
  {
    if (defined()) {
      return data_;
    }
    throw utils::runtime_error(*this, error_context("get_data"),
      "Attempted to access data of an undefined State");
  }

  /// @brief Get the variant storing the underlying engine (const).
  const StateData::VectorVariant& get_variant() const
  {
    return get_data()->vector_;
  }

  /// @brief Get the variant storing the underlying engine (mutable).
  StateData::VectorVariant& get_variant() { return get_data()->vector_; }

  /// @brief Set the label for this state.
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

  /// @brief Get the total number of elements represented by the state.
  int64_t get_numel() const
  {
    return std::visit(
      [](const auto& v) { return v.get_numel(); }, get_variant());
  }

  /// @brief Check if the state is represented by a Tensor Train engine.
  bool is_tt() const { return std::holds_alternative<TTEngine>(get_variant()); }

  /// @brief Retrieve the underlying TTEngine.
  /// @throws ttnte::utils::runtime_error If the state does not contain a
  /// TTEngine.
  const TTEngine& as_tt() const
  {
    if (auto* tt = std::get_if<TTEngine>(&get_variant()))
      return *tt;
    throw utils::runtime_error(
      *this, error_context("as_tt"), "State does not hold a TTEngine");
  }
};

// =================================================================
// Free Functions / Operators for States
// =================================================================

/// @brief Addition of a State and a scalar/tensor.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  State>::type
operator+(const State& a, const T& b)
{
  return std::visit(
    [&](const auto& v) -> State { return State(v + b); }, a.get_variant());
}

/// @brief Addition of a scalar/tensor and a State.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  State>::type
operator+(const T& a, const State& b)
{
  return b + a;
}

/// @brief Addition of two States.
inline State operator+(const State& a, const State& b)
{
  return std::visit(
    [](const auto& v0, const auto& v1) -> State { return State(v0 + v1); },
    a.get_variant(), b.get_variant());
}

/// @brief Subtraction of a scalar/tensor from a State.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  State>::type
operator-(const State& a, const T& b)
{
  return a + (-b);
}

/// @brief Subtraction of a State from a scalar/tensor.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  State>::type
operator-(const T& a, const State& b)
{
  return std::visit(
    [&](const auto& v) -> State { return State(a - v); }, b.get_variant());
}

/// @brief Subtraction of two States.
inline State operator-(const State& a, const State& b)
{
  return std::visit(
    [&](const auto& v0, const auto& v1) -> State { return State(v0 - v1); },
    a.get_variant(), b.get_variant());
}

/// @brief Multiplication of a State and a scalar/tensor.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  State>::type
operator*(const State& a, const T& b)
{
  return std::visit(
    [&](const auto& v) -> State { return State(v * b); }, a.get_variant());
}

/// @brief Multiplication of a scalar/tensor and a State.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  State>::type
operator*(const T& a, const State& b)
{
  return b * a;
}

/// @brief Multiplication of two States.
inline State operator*(const State& a, const State& b)
{
  return std::visit(
    [](const auto& v0, const auto& v1) -> State { return State(v0 * v1); },
    a.get_variant(), b.get_variant());
}

/// @brief Division of a State by a scalar/tensor.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  State>::type
operator/(const State& a, const T& b)
{
  return std::visit(
    [&](const auto& v) -> State { return State(v / b); }, a.get_variant());
}

/// @brief Division of a scalar/tensor by a State.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  State>::type
operator/(const T& a, const State& b)
{
  return std::visit(
    [&](const auto& v) -> State { return State(a / v); }, b.get_variant());
}

/// @brief Division of two States.
inline State operator/(const State& a, const State& b)
{
  return std::visit(
    [](const auto& v0, const auto& v1) -> State { return State(v0 / v1); },
    a.get_variant(), b.get_variant());
}

} // namespace ttnte::linalg
