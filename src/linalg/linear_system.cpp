#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/utils/exception.hpp"
#include <c10/core/DefaultDtype.h>
#include <torch/cuda.h>

namespace ttnte::linalg {

// =================================================================
// Protected constructors
LinearSystem::LinearSystem(Operator interior_op,
  c10::SmallVector<NeighborCoupling, 6> couplings, State state,
  Source::Ptr source, std::optional<std::string> label)
  : interior_op_(std::move(interior_op)), couplings_(std::move(couplings)),
    state_(std::move(state)), source_(std::move(source)),
    device_(interior_op_.get_device()),
    label_(label.has_value() ? Label::from_string(label.value())
                             : Label::create_internal())
{
  TORCH_CHECK(interior_op_.defined(), "The operators must be defined pointers");

  const auto dtype = interior_op_.get_dtype();
  auto options = torch::TensorOptions().dtype(dtype).device(device_);
  if (device_.is_cpu() && torch::cuda::is_available()) {
    options = options.pinned_memory(true);
  }

  // Compute flat buffer size: interior_op + boundary_ops + [state] + [source]
  int64_t total = interior_op_.get_numel();
  for (const auto& c : couplings_) {
    total += c.boundary_op.get_numel();
  }
  if (state_.defined()) {
    state_is_static_ = true;
    total += state_.get_numel();
    if (state_.get_device() != device_ || state_.get_dtype() != dtype) {
      throw utils::runtime_error(*this, error_context("LinearSystem"),
        "State must share device and dtype with the interior operator");
    }
  }
  if (source_ && source_->buffer_size() > 0) {
    total += source_->buffer_size();
    if (source_->get_device() != device_ || source_->get_dtype() != dtype) {
      throw utils::runtime_error(*this, error_context("LinearSystem"),
        "Source must share device and dtype with the interior operator");
    }
  }

  // Allocate contiguous pinned buffer and pack all static data into it
  torch::Tensor buffer = torch::empty({total}, options);
  int64_t offset = 0;
  auto pack = [&](auto& obj) {
    int64_t len = obj.get_numel();
    torch::Tensor sub = buffer.narrow(0, offset, len);
    obj.to_buffer(sub);
    obj.from_buffer(std::move(sub));
    offset += len;
  };

  pack(interior_op_);
  for (auto& c : couplings_)
    pack(c.boundary_op);
  if (state_.defined())
    pack(state_);
  if (source_ && source_->buffer_size() > 0) {
    const int64_t len = source_->buffer_size();
    torch::Tensor sub = buffer.narrow(0, offset, len);
    source_->to_buffer(sub);
    source_->from_buffer(sub);
    offset += len;
  }

  if (device_.is_cpu()) {
    host_buffer_ = std::move(buffer);
  } else if (device_.is_cuda()) {
    device_buffer_ = std::move(buffer);
  } else {
    std::stringstream ss;
    ss << "This device (" << device_ << ") is not supported";
    throw utils::runtime_error(*this, error_context("LinearSystem"), ss.str());
  }
}

// =================================================================
// Public methods
void LinearSystem::transfer_buffer(
  const torch::TensorOptions& options, bool non_blocking, bool copy)
{
  // Update the device if needed
  if (options.has_device()) {
    device_ = *options.device_opt();
  }

  torch::Tensor buffer;
  if (device_.is_cuda()) {
    if (!device_buffer_.defined()) {
      device_buffer_ = host_buffer_.to(options, non_blocking, copy);
    } else {
      device_buffer_ = device_buffer_.to(options, non_blocking, copy);
    }
    buffer = device_buffer_;

  } else {
    const auto pinned = [&](torch::TensorOptions o) {
      return torch::cuda::is_available() ? o.pinned_memory(true) : o;
    };

    if (!device_buffer_.defined()) {
      // CPU-only path: no GPU copy exists, apply dtype cast in place.
      host_buffer_ = host_buffer_.to(pinned(options), non_blocking, copy);

    } else if (!host_buffer_.defined()) {
      // First d2h from a GPU-constructed system.
      host_buffer_ = device_buffer_.to(pinned(options), non_blocking, copy);
      device_buffer_ = torch::Tensor();

    } else {
      // Subsequent d2h: buffer data (operators) is read-only so host_buffer_
      // is always in sync with device_buffer_. Only reallocate when dtype
      // changes; otherwise just free the GPU allocation.
      if (options.has_dtype() && options.dtype() != host_buffer_.dtype()) {
        host_buffer_ = device_buffer_.to(pinned(options), non_blocking, copy);
      }
      device_buffer_ = torch::Tensor();
    }
    buffer = host_buffer_;
  }

  int64_t offset = 0;
  auto unpack = [&](auto& obj) {
    int64_t len = obj.get_numel();
    obj.from_buffer(buffer.narrow(0, offset, len));
    offset += len;
  };

  unpack(interior_op_);
  for (auto& c : couplings_) {
    unpack(c.boundary_op);
  }
  if (state_is_static_)
    unpack(state_);
  if (source_ && source_->buffer_size() > 0) {
    const int64_t len = source_->buffer_size();
    source_->from_buffer(buffer.narrow(0, offset, len));
    offset += len;
  }
}

void LinearSystem::transfer_buffer(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  const auto options =
    torch::TensorOptions().device(device).dtype(dtype).memory_format(
      memory_format);

  transfer_buffer(options, non_blocking, copy);
}

void LinearSystem::transfer_buffer(const torch::Device& device,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  const auto options =
    torch::TensorOptions().device(device).memory_format(memory_format);

  transfer_buffer(options, non_blocking, copy);
}

void LinearSystem::transfer_nonbuffer(
  const torch::TensorOptions& options, bool non_blocking, bool copy)
{
  if (state_.defined() && !state_is_static_) {
    state_.to_(options, non_blocking, copy);
  }
  if (source_ && source_->is_eigenvalue()) {
    source_->transfer_nonbuffer(options, non_blocking, copy);
  }
  for (auto& c : couplings_) {
    if (c.send_buffer.defined()) {
      c.send_buffer.to_(options, non_blocking, copy);
    }
    if (c.recv_buffer.defined()) {
      c.recv_buffer.to_(options, non_blocking, copy);
    }
  }
}

void LinearSystem::transfer_nonbuffer(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  const auto options =
    torch::TensorOptions().device(device).dtype(dtype).memory_format(
      memory_format);

  transfer_nonbuffer(options, non_blocking, copy);
}

void LinearSystem::transfer_nonbuffer(const torch::Device& device,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  const auto options =
    torch::TensorOptions().device(device).memory_format(memory_format);

  transfer_nonbuffer(options, non_blocking, copy);
}

void LinearSystem::to_(
  const torch::TensorOptions& options, bool non_blocking, bool copy)
{
  transfer_buffer(options, non_blocking, copy);
  transfer_nonbuffer(options, non_blocking, copy);
}

void LinearSystem::to_(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  const auto options =
    torch::TensorOptions().device(device).dtype(dtype).memory_format(
      memory_format);

  this->to_(options, non_blocking, copy);
}

void LinearSystem::to_(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  const auto options =
    torch::TensorOptions().device(device).memory_format(memory_format);

  this->to_(options, non_blocking, copy);
}

void LinearSystem::to_(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  const auto options =
    torch::TensorOptions().dtype(dtype).memory_format(memory_format);

  this->to_(options, non_blocking, copy);
}

// =================================================================
// Public getters / setters

void LinearSystem::set_state(State state)
{
  if (state_is_static_) {
    throw utils::runtime_error(
      *this, error_context("set_state"), "This state vector was marked static");
  }

  // Compute the error between the two
  if (state_.defined() && state.defined()) {
    error_ = (state - state_).norm() / state_.norm();
  }

  state_ = std::move(state);
}

} // namespace ttnte::linalg
