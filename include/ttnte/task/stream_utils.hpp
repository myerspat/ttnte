#pragma once

#include "ttnte/linalg/state.hpp"
#include "ttnte/parallel/stream_handle.hpp"
#include "ttnte/task/task.hpp"
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace {

/// @brief Test an active CUDA stream to see if it has completed. The stream
/// itself is a thread's own permanently-claimed one (see
/// parallel::StreamPool::claim_for_this_thread()) -- unlike the old
/// shared-pool design, there is nothing to release back to a pool here.
/// @param active_stream The stream this task's work was issued on.
/// @return COMPLETED once the stream is drained and a fence has been
/// established; POLLING otherwise.
inline ttnte::task::TaskStatus test_stream(
  const ttnte::parallel::StreamHandle& active_stream)
{
  // Always query via the CUDA impl — all streams in our pool are CUDA streams,
  // even for d2h transfers whose target_device.type() is CPU. Passing CPU to
  // getDeviceGuardImpl always returns true (CPU has no async streams), causing
  // the next iteration's h2d to race the still-in-flight DMA into host_buffer_.
  bool is_done = c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
                   ->queryStream(active_stream.stream);

  if (is_done) {
    // cudaStreamQuery confirms the stream is empty, but does NOT establish a
    // GPU-wide memory fence. Without an explicit synchronize, kernels on a
    // subsequent task (issued on this same, thread-owned stream) may not see
    // writes committed by the completed work — even if queryStream returned
    // true. This matches what GPU_SYNC mode does via synchronizeStream, and is
    // instantaneous because the stream is already drained.
    c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
      ->synchronizeStream(active_stream.stream);
    return ttnte::task::TaskStatus::COMPLETED;

  } else {
    // The GPU is still working / data is still moving.
    return ttnte::task::TaskStatus::POLLING;
  }
}

/// @brief Tell the CUDA caching allocator that `stream` is also reading/
/// writing `s`'s underlying TT-core tensors, so it won't hand their memory
/// to an unrelated allocation on a different stream until `stream` has
/// caught up too. Needed because task dispatch is not sticky per patch --
/// the same State's tensors routinely get touched by a different worker's
/// stream than the one that wrote them (solve_task writes on whichever
/// stream drew that dispatch; a later narrow_task/apply_task/solve_task for
/// the same patch may run on a different one), and without this the
/// allocator only tracks the tensor's original allocating stream.
///
/// Shared between configure_cuda_task.hpp (compute tasks, always GPU) and
/// configure_cpu_task.hpp (boundary-exchange tasks, which are dispatched as
/// CPU_SYNC/CPU_ASYNC purely because they're host-orchestrated handoffs, but
/// whose buffers are GPU-resident whenever the boundary state stays on
/// device -- see configure_cpu_task.hpp's pack/unpack/within-rank-send
/// tasks for why they need this too).
inline void record_stream_state(
  const ttnte::linalg::State& s, c10::Stream stream)
{
  if (!s.defined()) {
    return;
  }
  for (const auto& core : s.as_tt().get_cores()) {
    core.record_stream(stream);
  }
}

} // namespace
