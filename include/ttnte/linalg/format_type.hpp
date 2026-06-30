#pragma once

#include <cstdint>

namespace ttnte::linalg {

/// @brief Enum class for the format of a linear algebra object.
enum class FormatType : uint8_t { DENSE = 0, TENSOR_TRAIN = 1 };

} // namespace ttnte::linalg
