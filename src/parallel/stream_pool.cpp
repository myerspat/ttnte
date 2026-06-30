#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/cuda.h>

namespace ttnte::parallel {

// =================================================================
// Private constructors
StreamPool::StreamPool(int num_streams)
  : device_(parallel::ParallelContext::instance().device())
{
  if (torch::cuda::is_available()) {
    // Retrieve the number of streams requested
    auto* guard_impl = c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA);

    streams_.reserve(num_streams);
    for (int i = 0; i < num_streams; i++) {
      streams_.emplace_back(
        guard_impl->getStreamFromGlobalPool(device_, false));
    }
  }
}

// =================================================================
// Public methods
StreamPool::Ptr StreamPool::instance(int num_streams)
{
  static Ptr pool = nullptr;
  static std::once_flag init_flag;

  std::call_once(
    init_flag,
    [](Ptr& pool, int num_streams) { pool = Ptr(new StreamPool(num_streams)); },
    pool, num_streams);

  return pool;
}

std::optional<StreamHandle> StreamPool::try_acquire()
{
  std::unique_lock<std::mutex> lock(mutex_);

  if (streams_.empty()) {
    return std::nullopt;
  }

  auto stream = streams_.back();
  streams_.pop_back();
  return stream;
}

void StreamPool::release(const StreamHandle& stream)
{
  std::unique_lock<std::mutex> lock(mutex_);
  streams_.push_back(std::move(stream));
}

} // namespace ttnte::parallel
