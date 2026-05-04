#pragma once

#include "ttnte/parallel/stream_guard.hpp"
#include <c10/core/Stream.h>

namespace ttnte::parallel {

/// @brief Wrapper class for torch GPU streams.
struct StreamHandle {
  /// The CUDA stream casted to the base class.
  c10::Stream stream;
  /// @return A StreamGuard for this stream.
  [[nodiscard]] StreamGuard guard() const { return StreamGuard(stream); }
};

} // namespace ttnte::parallel
