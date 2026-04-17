#pragma once

#include "ttnte/parallel/stream_guard.hpp"
#include <c10/core/Stream.h>

namespace ttnte::parallel {

struct StreamHandle {
  c10::Stream stream;
  [[nodiscard]] StreamGuard guard() const { return StreamGuard(stream); }
};

} // namespace ttnte::parallel
