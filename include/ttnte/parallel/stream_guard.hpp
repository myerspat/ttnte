#pragma once

#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>

namespace ttnte::parallel {

class StreamGuard {
private:
  // =================================================================
  // Private data
  c10::StreamGuard guard_;

public:
  // =================================================================
  // Public constructors
  explicit StreamGuard(c10::Stream stream) : guard_(stream) {}

  // Prevent copying or moving to maintain RAII safety
  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;
};

} // namespace ttnte::parallel
