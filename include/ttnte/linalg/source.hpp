#pragma once

#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/ops.hpp"
#include "ttnte/linalg/state.hpp"
#include <limits>
#include <memory>
#include <torch/types.h>

namespace ttnte::linalg {

// =================================================================
// Source

/// @brief Patch-local source term. The source State is static and goes
/// entirely into the LinearSystem flat buffer for a single DMA transfer to
/// device. Subclass with EigenSource for problems where the source depends on
/// the solution of the current iteration.
class Source {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<Source>;

protected:
  // =================================================================
  // Protected data
  State state_;

  // =================================================================
  // Protected constructors
  Source() = default;
  explicit Source(State state) : state_(std::move(state)) {}

public:
  virtual ~Source() = default;

  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new Source(std::forward<Args>(args)...));
  }

  /// @brief Recompute state_ from the current solution iterate.
  /// No-op for fixed sources; overridden by EigenSource.
  /// @param eigvec Current solution iterate.
  /// @param eps TT rounding tolerance.
  /// @param max_rank Maximum TT rank after rounding.
  virtual void update(const State& /*eigvec*/, double /*eps*/ = 1e-12,
    int64_t /*max_rank*/ = std::numeric_limits<int64_t>::max())
  {}

  /// @brief Transfer dynamic (non-buffer) data to another device or dtype.
  /// No-op for fixed sources (all data is in the flat buffer). Overridden by
  /// EigenSource to transfer state_ which is computed outside the buffer.
  /// @param options Target device and/or dtype.
  /// @param non_blocking Non-blocking transfer for pinned CPU memory.
  /// @param copy Force a copy even if options already match.
  virtual void transfer_nonbuffer(const torch::TensorOptions& options,
    bool non_blocking = false, bool copy = false)
  {}

  virtual void scale() {}

  /// @brief Copy the static source state into a contiguous buffer slice.
  /// @param buf Flat tensor slice of length buffer_size().
  virtual void to_buffer(const torch::Tensor& buf) { state_.to_buffer(buf); }
  /// @brief Restore tensor views from a buffer slice after a device transfer.
  /// @param buf Flat tensor slice of length buffer_size().
  virtual void from_buffer(const torch::Tensor& buf)
  {
    state_.from_buffer(buf);
  }
  /// @brief Number of scalar elements to_buffer / from_buffer will read/write.
  virtual int64_t buffer_size() const { return state_.get_numel(); }

  // =================================================================
  // Public getters / setters
  /// @return True if this source depends on the current solution iterate.
  virtual bool is_eigenvalue() const noexcept { return false; }
  /// @return Device of the static data packed into the flat buffer.
  virtual torch::Device get_device() const { return state_.get_device(); }
  /// @return Scalar type of the static data packed into the flat buffer.
  virtual at::ScalarType get_dtype() const { return state_.get_dtype(); }
  /// @return The source state vector.
  const State& get_state() const noexcept { return state_; }
  /// @param state The new source state vector.
  void set_state(State state) { state_ = std::move(state); }
};

// =================================================================
// EigenSource

/// @brief Source term for a generalized eigenvalue problem. The operator op_
/// is static and packed into the LinearSystem flat buffer for a single DMA
/// transfer to device. The source state (state_ = op_ * eigvec) is dynamic —
/// recomputed each iteration by a DAG task via update(). eigval_ is the scale
/// factor written by the driver before each DAG execution, allowing the DAG to
/// be built once and reused across iterations.
class EigenSource : public Source {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<EigenSource>;

protected:
  // =================================================================
  // Protected data
  /// Static operator packed into the flat buffer.
  Operator op_;
  /// Scale factor applied to the source state. Updated by the driver each
  /// iteration before DAG execution.
  double eigval_ = 1.0;
  /// Sum of state_, cached after each update() call.
  double total_source_ = 0.0;

  // =================================================================
  // Protected constructors
  explicit EigenSource(Operator op) : Source(), op_(std::move(op)) {}

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new EigenSource(std::forward<Args>(args)...));
  }

  /// @brief Pack the operator into the buffer (state_ is dynamic and excluded).
  void to_buffer(const torch::Tensor& buf) override final
  {
    op_.to_buffer(buf);
  }
  /// @brief Restore the operator's tensor views from the buffer.
  void from_buffer(const torch::Tensor& buf) override final
  {
    op_.from_buffer(buf);
  }
  /// @return Get the size of the static buffer needed to finalize this eigen
  /// source.
  int64_t buffer_size() const override final { return op_.get_numel(); }
  /// @return Is this an eigenvalue source?
  bool is_eigenvalue() const noexcept override final { return true; }

  /// @brief Transfer the dynamic source state (state_ = op_ * eigvec) to
  /// another device or dtype so it follows the non-buffer state transfers.
  void transfer_nonbuffer(const torch::TensorOptions& options,
    bool non_blocking = false, bool copy = false) override final
  {
    if (state_.defined())
      state_.to_(options, non_blocking, copy);
  }

  /// @brief Recompute state_ = mv(op_, eigvec) and cache the element sum.
  /// Called by the update DAG task after each local solve.
  /// @param eigvec The current solution iterate.
  /// @param eps TT rounding tolerance applied after the mv.
  /// @param max_rank Maximum TT rank after rounding.
  void update(const State& eigvec, double eps = 1e-12,
    int64_t max_rank = std::numeric_limits<int64_t>::max()) override final
  {
    // Apply operator to the eigenvector
    state_ = mv(op_, eigvec.to(op_.get_device())).to(eigvec.get_device());
    state_.round_(eps, max_rank);

    // Compute the total source
    total_source_ = state_.sum();
  }

  /// @brief Scale the eigen source vector by the eigenvalue.
  void scale() override final { state_ *= eigval_; }

  // =================================================================
  // Public getters / setters
  /// @return The operator (buffer-backed after a device transfer).
  const Operator& get_op() const noexcept { return op_; }
  /// @return Sum of state_, cached by the most recent update() call.
  double get_total_source() const noexcept { return total_source_; }

  /// @brief Set the scale factor. Called by the driver before each DAG
  /// execution so scale tasks read the current value at execution time.
  /// @param eigval The new scale factor.
  void set_eigval(double eigval) noexcept { eigval_ = eigval; }
  /// @return The current eigenvalue.
  double get_eigval() const noexcept { return eigval_; }
  /// @return The device this source is on.
  torch::Device get_device() const override final { return op_.get_device(); }
  /// @return The data type of the source.
  at::ScalarType get_dtype() const override final { return op_.get_dtype(); }
};

} // namespace ttnte::linalg
