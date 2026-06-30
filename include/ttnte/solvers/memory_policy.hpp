#pragma once

#include <cstdint>

namespace ttnte::solvers {

/// @brief The memory policy for what devices data should persist.
enum class MemoryPolicy : uint8_t {
  RESIDENT,          /// Operators and state vectors remain on the GPU
  STATE_RESIDENT,    /// Only the state vectors remain on the GPU while the
                     /// operators are removed between solves
  OPERATOR_RESIDENT, /// Only the operators persist on the GPU
  OUT_OF_CORE /// Neither operators nor state vectors persist in GPU memory
};

} // namespace ttnte::solvers
