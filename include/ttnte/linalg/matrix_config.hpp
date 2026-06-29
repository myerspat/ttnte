#pragma once

#include <cstdint>

namespace ttnte::linalg {

/// @brief Enum class for defining a portion of a matrix.
enum MatrixComponent : uint8_t {
  /// No part of the matrix
  NONE = 0,
  /// Only the diagonal part.
  DIAGONAL = 1 << 0,
  /// Only the lower triangular part.
  LOWER = 1 << 1,
  /// Only the upper triangular part.
  UPPER = 1 << 2,
  /// The lower triangular part including the diagonal.
  LOWER_INCLUSIVE = DIAGONAL | LOWER,
  /// The upper triangular part including the diagonal.
  UPPER_INCLUSIVE = DIAGONAL | UPPER,
  /// The full matrix.
  FULL = DIAGONAL | LOWER | UPPER,
};

} // namespace ttnte::linalg
