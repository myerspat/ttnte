#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::linalg {

// =================================================================
// Public constructors
TTEngine::TTEngine(const Tensors& cores, bool check_cores) : cores_(cores)
{
  if (check_cores) {
    if (cores.empty()) {
      throw utils::runtime_error(
        "ttnte::linalg::TTEngine::TTEngine", "`cores` is empty");
    }

    // Check the cores are all on the same device with the same data type
    const auto& device = cores_[0].device();
    const auto& dtype = cores_[0].dtype();

    // Iterate through each core and make sure they're the right shape
    int64_t left_rank = 1;
    for (auto& core : cores_) {
      if (core.device() != device || core.dtype() != dtype) {
        throw utils::runtime_error("ttnte::linalg::TTEngine::TTEngine",
          "All TT-cores must be on the same device with the same data type");
      }

      if (core.ndimension() != 3 && core.ndimension() != 4) {
        throw utils::runtime_error("ttnte::linalg::TTEngine::TTEngine",
          "All TT-cores must either be 3-dimensional for vectors or\n"
          "4-dimensional for matrices");
      }

      if (core.size(0) != left_rank) {
        throw utils::runtime_error("ttnte::linalg::TTEngine::TTEngine",
          "The right rank of core k must equal the left rank of core k + 1");
      }

      if (core.ndimension() == 3) {
        core = core.reshape({core.size(0), core.size(1), 1, core.size(2)});
      }
      core = core.contiguous();

      left_rank = core.size(3);
    }

    if (left_rank != 1) {
      throw utils::runtime_error("ttnte::linalg::TTEngine::TTEngine",
        "The last rank of the last core must equal to 1");
    }

  } else {
    cores_ = cores;
  }
}

// =================================================================
// Public methods
TTEngine TTEngine::clone_from(const Tensors& cores, bool check_cores)
{
  Tensors new_cores;
  new_cores.reserve(cores.size());

  for (const auto& core : cores) {
    new_cores.push_back(core.clone());
  }
  return TTEngine(std::move(new_cores), check_cores);
}

TTEngine TTEngine::to(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  Tensors new_cores;
  new_cores.reserve(cores_.size());

  for (const auto& core : cores_) {
    new_cores.push_back(
      core.to(device, dtype, non_blocking, copy, memory_format));
  }

  return TTEngine(std::move(new_cores));
}

TTEngine TTEngine::to(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return to(cores_[0].device(), dtype, non_blocking, copy, memory_format);
}
TTEngine TTEngine::to(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return to(device, cores_[0].scalar_type(), non_blocking, copy, memory_format);
}

TTEngine& TTEngine::to_(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  for (auto& core : cores_) {
    core = core.to(device, dtype, non_blocking, copy, memory_format);
  }
  return *this;
}

TTEngine& TTEngine::to_(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  return to_(cores_[0].device(), dtype, non_blocking, copy, memory_format);
}
TTEngine& TTEngine::to_(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  return to_(
    device, cores_[0].scalar_type(), non_blocking, copy, memory_format);
}

void TTEngine::from_buffer(const torch::Tensor& buffer)
{
  assert(buffer.numel() == get_numel() && buffer.ndimension() == 1);

  int64_t idx = 0;
  for (auto& core : cores_) {
    const auto& shape = core.sizes();
    core = buffer.narrow(0, idx, core.numel()).view(shape);
    idx += core.numel();
  }
}

// =================================================================
// Public getters / setters
int64_t TTEngine::get_numel() const
{
  int64_t total = 0;
  for (const auto& core : cores_) {
    total += core.numel();
  }
  return total;
}

c10::SmallVector<int64_t, 5> TTEngine::get_ranks() const
{
  c10::SmallVector<int64_t, 5> ranks;
  ranks.reserve(cores_.size() - 1);

  for (auto it = cores_.begin(); it != cores_.end() - 1; it++) {
    ranks.push_back(it->size(3));
  }

  return ranks;
}

c10::SmallVector<int64_t, 12> TTEngine::get_free_indices() const
{
  c10::SmallVector<int64_t, 12> free_indices;
  free_indices.reserve(2 * cores_.size());

  for (const auto& core : cores_) {
    free_indices.push_back(core.size(1));
    free_indices.push_back(core.size(2));
  }

  return free_indices;
}

} // namespace ttnte::linalg
