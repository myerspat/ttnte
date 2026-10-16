#pragma once

#include <cstdint>
#include <vector>

namespace ttnte::sweep {

/// @brief A block of fibers to sweep: every pair of `directions` x
/// `groups`, each returning the angular flux over all local dofs.
struct FiberBlock {
  std::vector<int64_t> directions;
  std::vector<int64_t> groups;
};

} // namespace ttnte::sweep
