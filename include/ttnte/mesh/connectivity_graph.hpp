#pragma once

#include <torch/extension.h>

namespace ttnte::mesh {

struct ConnectivityGraph {
  /// The global IDs (GIDs) for the local (on this MPI rank) MeshBlocks
  torch::Tensor local_gids;
  /// The integer ranges for each GID neighbor in adjncy
  torch::Tensor xadj;
  /// The neighbors GID where block local_gids[i] has the neighbors
  /// adjncy[xadj[i]:xajd[i + 1]]
  torch::Tensor adjncy;
  /// The rank that each of the ID in the adjncy lives
  torch::Tensor mpi_ranks;
};

} // namespace ttnte::mesh
