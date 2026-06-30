#include "ttnte/parallel/parmetis_ops.hpp"
#include "ttnte/utils/exception.hpp"
#include <metis.h>
#include <parmetis.h>

namespace ttnte::parallel {

torch::Tensor kway_partition(int64_t nvtxs, const torch::Tensor& xadj,
  const torch::Tensor& adjncy, const torch::Tensor& vwgt, int32_t ncon,
  const torch::Tensor& ubvec, const Communicator& comm)
{
  // Get data type of idx_t and real_t
  const auto idx_dtype = (sizeof(idx_t) == 4) ? torch::kInt32 : torch::kInt64;
  const auto real_dtype =
    (sizeof(real_t) == 4) ? torch::kFloat32 : torch::kFloat64;

  // Force all input tensors into the exact types and contiguous memory
  torch::Tensor safe_xadj = xadj.to(idx_dtype).contiguous();
  torch::Tensor safe_adjncy = adjncy.to(idx_dtype).contiguous();
  torch::Tensor safe_vwgt = vwgt.to(idx_dtype).contiguous();
  torch::Tensor safe_ubvec = ubvec.to(real_dtype).contiguous();

  // Create output tensor
  torch::Tensor part =
    torch::zeros({xadj.size(0) - 1}, torch::dtype(idx_dtype));

  idx_t nvtxs_idx = static_cast<idx_t>(nvtxs);
  idx_t ncon_idx = static_cast<idx_t>(ncon);
  idx_t nparts = comm.size();
  idx_t objval;

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;

  auto status = METIS_PartGraphKway(&nvtxs, &ncon_idx,
    safe_xadj.data_ptr<idx_t>(), safe_adjncy.data_ptr<idx_t>(),
    safe_vwgt.data_ptr<idx_t>(), NULL, NULL, &nparts, NULL,
    safe_ubvec.data_ptr<real_t>(), options, &objval, part.data_ptr<idx_t>());

  if (status != METIS_OK) {
    std::string err_msg =
      "METIS_PartGraphKway failed with status: " + std::to_string(status);
    if (status == METIS_ERROR_INPUT)
      err_msg += " (Invalid Input Graph/Weights)";
    if (status == METIS_ERROR_MEMORY)
      err_msg += " (Out of Memory)";
    if (status == METIS_ERROR)
      err_msg += " (Unknown Internal Error)";

    throw utils::runtime_error("ttnte::parallel::kway_partition", err_msg);
  }

  return part.to(torch::kInt64);
}

torch::Tensor adaptive_repart(const torch::Tensor& vtxdist,
  const torch::Tensor& xadj, const torch::Tensor adjncy,
  const torch::Tensor& vwgt, int32_t ncon, const torch::Tensor& tpwgts,
  const torch::Tensor& ubvec, const Communicator& comm)
{
  // Get data type of idx_t and real_t
  const auto idx_dtype = (sizeof(idx_t) == 4) ? torch::kInt32 : torch::kInt64;
  const auto real_dtype =
    (sizeof(real_t) == 4) ? torch::kFloat32 : torch::kFloat64;

  // Force all input tensors into the exact types and contiguous memory
  torch::Tensor safe_vtxdist = vtxdist.to(idx_dtype).contiguous();
  torch::Tensor safe_xadj = xadj.to(idx_dtype).contiguous();
  torch::Tensor safe_adjncy = adjncy.to(idx_dtype).contiguous();
  torch::Tensor safe_vwgt = vwgt.to(idx_dtype).contiguous();
  torch::Tensor safe_tpwgts = tpwgts.to(real_dtype).contiguous();
  torch::Tensor safe_ubvec = ubvec.to(real_dtype).contiguous();

  // Create output tensor
  torch::Tensor part =
    torch::zeros({xadj.size(0) - 1}, torch::dtype(idx_dtype));

  // ParMETIS setup flags
  idx_t wgtflag = 2; // Vertex weights are present
  idx_t numflag = 0; // 0-indexed C-arrays
  idx_t nparts = comm.size();
  idx_t options[4] = {0, 0, 0, 0};
  real_t ipc2redist = 1000.0;
  idx_t edgecut;
  idx_t ncon_idx = static_cast<idx_t>(ncon);
  MPI_Comm c = MPI_Comm_f2c(comm.get());

  auto status = ParMETIS_V3_AdaptiveRepart(safe_vtxdist.data_ptr<idx_t>(),
    safe_xadj.data_ptr<idx_t>(), safe_adjncy.data_ptr<idx_t>(),
    safe_vwgt.data_ptr<idx_t>(), NULL, NULL, &wgtflag, &numflag, &ncon_idx,
    &nparts, safe_tpwgts.data_ptr<real_t>(), safe_ubvec.data_ptr<real_t>(),
    &ipc2redist, options, &edgecut, part.data_ptr<idx_t>(), &c);

  if (status != METIS_OK) {
    std::string err_msg = "ParMETIS_V3_AdaptiveRepart failed with status: " +
                          std::to_string(status);
    if (status == METIS_ERROR_INPUT)
      err_msg += " (Invalid Input Graph/Weights)";
    if (status == METIS_ERROR_MEMORY)
      err_msg += " (Out of Memory)";
    if (status == METIS_ERROR)
      err_msg += " (Unknown Internal Error)";

    throw utils::runtime_error("ttnte::parallel::adaptive_repart", err_msg);
  }

  return part.to(torch::kInt64);
}

} // namespace ttnte::parallel
