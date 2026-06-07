#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/utils/exception.hpp"
#include <c10/core/DefaultDtype.h>
#include <numeric>
#include <torch/cuda.h>

namespace ttnte::linalg {

// =================================================================
// Protected constructors
LinearSystem::LinearSystem(Operator interior_op, State state, State source,
  std::optional<std::string> label)
  : device_(interior_op_.get_device()), interior_op_(std::move(interior_op)),
    state_(std::move(state)), source_(std::move(source)),
    label_(label.has_value() ? Label::from_string(label.value())
                             : Label::create_internal())
{
  TORCH_CHECK(interior_op.defined(), "The operators must be defined pointers");

  // Get base configuration
  const auto dtype = interior_op_.get_dtype();
  auto options = torch::TensorOptions().dtype(dtype).device(device_);

  if (device_.is_cpu() && torch::cuda::is_available()) {
    options = options.pinned_memory(true);
  }

  // Get the size needed for the buffer, only that given to the constructor is
  // put into the buffer
  std::array<int64_t, 3> sizes = {interior_op_.get_numel(), 0, 0};
  if (state_.defined()) {
    state_is_static_ = true;
    sizes[1] = state_.get_numel();

    if (state.get_device() != device_ || state.get_dtype() != dtype) {
      throw utils::runtime_error(*this, error_context("LinearSystem"),
        "Operators, the state vector (if given), and the source vector (if\n"
        "given) must be on the same device with the same data type");
    }
  }
  if (source.defined()) {
    source_is_static_ = true;
    sizes[2] = source_.get_numel();

    if (source_.get_device() != device_ || source_.get_dtype() != dtype) {
      throw utils::runtime_error(*this, error_context("LinearSystem"),
        "Operators, the state vector (if given), and the source vector (if\n"
        "given) must be on the same device with the same data type");
    }
  }

  // Create the buffer and fill it
  torch::Tensor buffer =
    torch::empty({std::accumulate(sizes.cbegin(), sizes.cend(), 0)}, options);

  int64_t offset = 0;
  torch::Tensor subbuffer = buffer.narrow(0, offset, sizes[0]);
  interior_op_.to_buffer(subbuffer);
  interior_op_.from_buffer(std::move(subbuffer));
  offset += sizes[0];
  if (state_.defined()) {
    subbuffer = buffer.narrow(0, offset, sizes[1]);
    state_.to_buffer(subbuffer);
    state_.from_buffer(std::move(subbuffer));
    offset += sizes[1];
  }
  if (source_.defined()) {
    subbuffer = buffer.narrow(0, offset, sizes[2]);
    source_.to_buffer(subbuffer);
    source_.from_buffer(std::move(subbuffer));
    offset += sizes[2];
  }

  // Set the buffer
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
    if (!host_buffer_.defined()) {
      host_buffer_ =
        device_buffer_.to(options.pinned_memory(true), non_blocking, copy);
    } else {
      host_buffer_ = host_buffer_.to(options, non_blocking, copy);
    }
    buffer = host_buffer_;
  }

  int64_t offset = 0;
  int64_t length = interior_op_.get_numel();
  interior_op_.from_buffer(buffer.narrow(0, offset, length));
  offset += length;

  if (state_is_static_) {
    length = state_.get_numel();
    state_.from_buffer(buffer.narrow(0, offset, length));
    offset += length;
  }

  if (source_is_static_) {
    length = source_.get_numel();
    source_.from_buffer(buffer.narrow(0, offset, length));
    offset += length;
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
  // Transfer non-static members
  if (state_.defined() && !state_is_static_) {
    state_.to_(options, non_blocking, copy);
  }
  if (source_.defined() && !source_is_static_) {
    source_.to_(options, non_blocking, copy);
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

  state_ = std::move(state);
}

void LinearSystem::set_source(State source)
{
  if (source_is_static_) {
    throw utils::runtime_error(
      *this, error_context("set_state"), "This state vector was marked static");
  }

  source_ = std::move(source);
}

} // namespace ttnte::linalg
