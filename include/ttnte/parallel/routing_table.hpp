#pragma once

#include <unordered_map>
#include <vector>

namespace ttnte::parallel {

/// @brief Struct for which MPI ranks should own what.
struct RoutingTable {
  /// Map target rank to a list of GIDs
  std::unordered_map<int, std::vector<int64_t>> send_blocks;
  /// Map source rank to a list of GIDs
  std::unordered_map<int, std::vector<int64_t>> recv_blocks;
};

} // namespace ttnte::parallel
