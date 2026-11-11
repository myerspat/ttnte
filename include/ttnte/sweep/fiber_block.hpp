#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace ttnte::sweep {

/// @brief A block of fibers to sweep: every pair of `directions` x
/// `groups`, each returning the angular flux over all local dofs.
struct FiberBlock {
  std::vector<int64_t> directions;
  std::vector<int64_t> groups;
};

/// @brief Called once for every range of finished output rows of a fiber
/// request, `count` rows from `first`, possibly from several worker threads
/// at once (each range belongs to one caller): work on finished fibers runs
/// while other fibers are still being swept.
using FiberDone = std::function<void(int64_t first, int64_t count)>;

} // namespace ttnte::sweep
