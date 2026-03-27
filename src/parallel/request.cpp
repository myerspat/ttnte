#include "ttnte/parallel/request.hpp"
#include <mpi.h>

namespace ttnte::parallel {

void Request::wait()
{
  MPI_Request c_req = MPI_Request_f2c(f_req_);
  MPI_Wait(&c_req, MPI_STATUS_IGNORE);
  f_req_ = MPI_Request_c2f(
    c_req); // MPI_Wait sets the request to NULL, so we update our int
}

void Request::wait_all(std::vector<Request>& requests)
{
  if (requests.empty())
    return;

  // Convert our safe ints back into C-style MPI_Requests
  std::vector<MPI_Request> c_reqs(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    c_reqs[i] = MPI_Request_f2c(requests[i].get());
  }

  // Do the highly-optimized bulk wait
  MPI_Waitall(c_reqs.size(), c_reqs.data(), MPI_STATUSES_IGNORE);

  // Update our C++ objects so they know they are finished (set to NULL)
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i].update(MPI_Request_c2f(c_reqs[i]));
  }
}

} // namespace ttnte::parallel
