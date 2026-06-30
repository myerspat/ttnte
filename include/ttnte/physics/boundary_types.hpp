#pragma once

#include <c10/util/SmallVector.h>
#include <cstdint>
#include <string_view>

namespace ttnte::physics {

/// @brief The boundary condition types for transport.
enum class BoundaryType : uint8_t {
  UNKNOWN,
  INTERNAL,
  VACUUM,
  REFLECTIVE,
  DEGENERATE,
};

/// @brief Convert the BoundaryType to a string name.
/// @param type The BoundaryType.
/// @return The type as a string.
[[nodiscard]] constexpr std::string_view to_string(BoundaryType type)
{
  switch (type) {
  case BoundaryType::INTERNAL:
    return "INTERNAL";
  case BoundaryType::VACUUM:
    return "VACUUM";
  case BoundaryType::REFLECTIVE:
    return "REFLECTIVE";
  case BoundaryType::DEGENERATE:
    return "DEGENERATE";
  default:
    return "UNKNOWN";
  }
}

/// @brief Which planes to apply a prescribed boundary condition type to.
struct BCPlane {
private:
  /// A vector of active planes where the boundary condition is applied. This is
  /// assumed to be ordered as [x_min, x_max, y_min, y_max, ...]. If, for
  /// example, active_planes_[0] == true then the prescribed boundary condition
  /// is applied to all axis-aligned boundaries that fall on the minimum x
  /// coordinate in the bounding box of the mesh.
  c10::SmallVector<bool, 6> active_planes_;

public:
  BCPlane(bool x_min = false, bool x_max = false, bool y_min = false,
    bool y_max = false, bool z_min = false, bool z_max = false)
  {
    active_planes_.push_back(x_min);
    active_planes_.push_back(x_max);
    active_planes_.push_back(y_min);
    active_planes_.push_back(y_max);
    active_planes_.push_back(z_min);
    active_planes_.push_back(z_max);
  }
  BCPlane(const c10::SmallVector<bool, 6>& active_planes)
    : active_planes_(active_planes)
  {}

  const c10::SmallVector<bool, 6>& get_active_planes() const noexcept
  {
    return active_planes_;
  }
};

} // namespace ttnte::physics
