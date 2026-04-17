#include "ttnte/linalg/tt_linear_system.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/utils/exception.hpp"
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <cstdint>

namespace ttnte::linalg {

// =================================================================
// Protected constructor
TTLinearSystem::TTLinearSystem(const OpPtr& interior_op, const StPtr& state,
  const StPtr& source, std::optional<std::string> label)
  : LinearSystem(label),
    state_(state->to(interior_op->get_device(), interior_op->get_dtype())),
    source_(source->to(interior_op->get_device(), interior_op->get_dtype()))
{
  // Get cores to pack
  const auto& int_cores = interior_op->get_cores();
  const auto& st_cores = state->get_cores();
  const auto& so_cores = source->get_cores();

  if (int_cores.size() != st_cores.size() ||
      int_cores.size() != so_cores.size()) {
    throw utils::runtime_error(*this, error_context("TTLinearSystem"),
      "There must be the same number of cores between the operators, source\n"
      "vector, and solution vector");
  }

  // Flatten the cores
  TTEngine::Tensors new_cores;
  new_cores.reserve(int_cores.size());
  for (size_t i = 0; i < int_cores.size(); i++) {
    const auto& core = int_cores[i];
    new_cores.push_back(core.flatten());

    // Check the shapes match the source and state vectors
    if (core.size(1) != so_cores[i].size(1) ||
        core.size(2) != st_cores[i].size(1)) {
      throw utils::runtime_error(*this, error_context("TTLinearSystem"),
        "The size of the first dimension of each core in the operator must\n"
        "equal that in the source and the second dimension must match the\n"
        "state vector");
    }
  }

  torch::Tensor active_buffer;
  if (interior_op->is_cuda()) {
    host_buffer_ = torch::Tensor();
    device_buffer_ = torch::cat(new_cores, 0);
    active_buffer = device_buffer_;
  } else {
    host_buffer_ = torch::cat(new_cores, 0).pin_memory();
    device_buffer_ = torch::Tensor();
    active_buffer = host_buffer_;
  }

  // Create a "new" operator using the views into this buffer
  int64_t idx = 0;
  for (size_t i = 0; i < int_cores.size(); i++) {
    new_cores[i] = active_buffer.narrow(0, idx, int_cores[i].numel())
                     .view(int_cores[i].sizes());
    idx += int_cores[i].numel();
  }
  interior_op_ = TTOperator::create(new_cores, interior_op->get_label());

  is_finalized_ = true;
}

// LinearSystem::Ptr TTLinearSystem::to_impl(const torch::Device& device,
//   const at::ScalarType& dtype, bool non_blocking, bool copy,
//   std::optional<at::MemoryFormat> memory_format) const
// {}
//
// LinearSystem::Ptr TTLinearSystem::to_impl(const at::ScalarType& dtype,
//   bool non_blocking, bool copy,
//   std::optional<at::MemoryFormat> memory_format) const
// {}

// =================================================================
// Public methods
void TTLinearSystem::to_(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  transfer_buffer(device, dtype, non_blocking, copy, memory_format);
  transfer_nonbuffer(device, dtype, non_blocking, copy, memory_format);
}

void TTLinearSystem::to_(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  transfer_buffer(device, non_blocking, copy, memory_format);
  transfer_nonbuffer(device, non_blocking, copy, memory_format);
}

void TTLinearSystem::to_(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  transfer_buffer(
    interior_op_->get_device(), dtype, non_blocking, copy, memory_format);
  transfer_nonbuffer(
    state_->get_device(), dtype, non_blocking, copy, memory_format);
}

// TTLinearSystem::Ptr TTLinearSystem::to(const torch::Device& device,
//   const at::ScalarType& dtype, bool non_blocking, bool copy,
//   std::optional<at::MemoryFormat> memory_format)
// {}
//
// TTLinearSystem::Ptr TTLinearSystem::to(const torch::Device& device,
//   bool non_blocking, bool copy, std::optional<at::MemoryFormat>
//   memory_format)
// {}
//
// TTLinearSystem::Ptr TTLinearSystem::to(const at::ScalarType& dtype,
//   bool non_blocking, bool copy, std::optional<at::MemoryFormat>
//   memory_format)
// {}

void TTLinearSystem::transfer_buffer(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  // Transfer the buffer
  LinearSystem::transfer_buffer(
    device, dtype, non_blocking, copy, memory_format);

  // Update the operators using the active buffer
  interior_op_->from_buffer(get_buffer(device));
}

void TTLinearSystem::transfer_buffer(const torch::Device& device,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  // Transfer the buffer
  LinearSystem::transfer_buffer(device, non_blocking, copy, memory_format);

  // Update the operators using the active buffer
  interior_op_->from_buffer(get_buffer(device));
}

void TTLinearSystem::transfer_nonbuffer(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  // Send non-buffer objects (state and source) to the device
  state_->to_(device, dtype, non_blocking, copy, memory_format);
  source_->to_(device, dtype, non_blocking, copy, memory_format);
}

void TTLinearSystem::transfer_nonbuffer(const torch::Device& device,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  // Send non-buffer objects (state and source) to the device
  state_->to_(device, non_blocking, copy, memory_format);
  source_->to_(device, non_blocking, copy, memory_format);
}

} // namespace ttnte::linalg
