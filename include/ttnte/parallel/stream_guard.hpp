#pragma once

#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>

namespace ttnte::parallel {

/// @brief A wrapper class for torch CUDA stream guards.
class StreamGuard {
private:
  // =================================================================
  // Private data
  /// The stream guard casted as the base class.
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
