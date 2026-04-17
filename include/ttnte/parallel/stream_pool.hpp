#pragma once

#include "ttnte/parallel/stream_handle.hpp"
#include <c10/util/SmallVector.h>
#include <memory>
#include <optional>

namespace ttnte::parallel {

class StreamPool {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<StreamPool>;

private:
  // =================================================================
  // Private data
  c10::SmallVector<StreamHandle, 16> streams_;
  std::mutex mutex_;

  // =================================================================
  // Private constructors
  StreamPool(int num_streams = 16);

public:
  ~StreamPool() = default;

  // Prevent copying
  StreamPool(const StreamPool&) = delete;
  StreamPool& operator=(const StreamPool&) = delete;

  // =================================================================
  // Public methods
  static Ptr instance(int num_streams = 16);
  std::optional<StreamHandle> try_acquire();
  void release(const StreamHandle& stream);
};

} // namespace ttnte::parallel
