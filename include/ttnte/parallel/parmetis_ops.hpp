#pragma once

#include "ttnte/parallel/communicator.hpp"
#include <torch/extension.h>

namespace ttnte::parallel {

/// @brief This method is a pass though to METIS_PartGraphKway from METIS
/// (https://github.com/KarypisLab/METIS/tree/master). This computes a partition
/// of which MPI rank a mesh block or element should go.
/// @param nvtxs The number of blocks or elements.
/// @param xadj An index into adjncy for which elements correspond to which mesh
/// blocks. The blocks are assumed to be ordered from 0 to nvtxs - 1.
/// @param vwgt The target weights for each MPI rank after partitioning.
/// @param ncon The number of constraints optimized based on the weights.
/// @param ubvec The acceptable error tolerance for each MPI rank load.
/// @param comm The communicator for MPI_COMM_WORLD.
torch::Tensor kway_partition(int64_t nvtxs, const torch::Tensor& xadj,
  const torch::Tensor& adjncy, const torch::Tensor& vwgt, int32_t ncon,
  const torch::Tensor& ubvec, const Communicator& comm);

/// @brief This method is a pass through to ParMETIS_V3_AdaptiveRepart from
/// ParMETIS (https://github.com/KarypisLab/ParMETIS). This computes the
/// repartition following adaptive mesh refinement or assembly. Refer to
/// https://karypis.github.io/glaros/files/sw/parmetis/manual.pdf for what each
/// parameter is.
torch::Tensor adaptive_repart(const torch::Tensor& vtxdist,
  const torch::Tensor& xadj, const torch::Tensor adjncy,
  const torch::Tensor& vwgt, int32_t ncon, const torch::Tensor& tpwgts,
  const torch::Tensor& ubvec, const Communicator& comm);

} // namespace ttnte::parallel
