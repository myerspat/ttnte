#pragma once

#include "ttnte/mesh/mesh_block_boundary.hpp"

namespace ttnte::linalg {

template<typename OpType>
struct NeighborCoupling {
  /// Neighbor information for this coupled connection.
  std::shared_ptr<mesh::NeighborInfo> connection;
  /// The boundary operator coupling this block to the connected one.
  std::shared_ptr<OpType> boundary_op;

  // TODO: Add a buffer here for receiving boundary information from
};

} // namespace ttnte::linalg
