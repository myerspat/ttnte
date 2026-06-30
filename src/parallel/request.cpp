#include "ttnte/parallel/request.hpp"
#include <mpi.h>

namespace ttnte::parallel {

void Request::wait()
{
  if (f_req_ == 0)
    return;
  MPI_Request c_req = MPI_Request_f2c(f_req_);
  if (c_req == MPI_REQUEST_NULL)
    return;
  MPI_Wait(&c_req, MPI_STATUS_IGNORE);
  f_req_ = MPI_Request_c2f(c_req);
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

bool Request::test()
{
  if (f_req_ == 0)
    return true;
  MPI_Request c_req = MPI_Request_f2c(f_req_);
  if (c_req == MPI_REQUEST_NULL)
    return true;
  int flag = 0;

  // MPI_Test sets flag to true (non-zero) if the operation has completed
  MPI_Test(&c_req, &flag, MPI_STATUS_IGNORE);

  // If completed, c_req is automatically set to MPI_REQUEST_NULL.
  // We must update our internal type-erased integer to reflect this.
  if (flag) {
    f_req_ = MPI_Request_c2f(c_req);
  }

  return flag != 0;
}

void Request::test_some(
  std::vector<Request>& requests, std::vector<int>& completed_indices)
{
  completed_indices.clear(); // Ensure we start with an empty list

  if (requests.empty())
    return;

  int count = static_cast<int>(requests.size());

  // Convert our safe ints back into C-style MPI_Requests
  std::vector<MPI_Request> c_reqs(count);
  for (int i = 0; i < count; ++i) {
    c_reqs[i] = MPI_Request_f2c(requests[i].get());
  }

  int outcount = 0;
  std::vector<int> raw_indices(
    count); // Pre-allocate to max possible completed requests

  // MPI_Testsome checks all requests and returns how many finished (outcount),
  // along with their original indices in the raw_indices array.
  MPI_Testsome(
    count, c_reqs.data(), &outcount, raw_indices.data(), MPI_STATUSES_IGNORE);

  // MPI_UNDEFINED means there were no active requests (all were already NULL)
  if (outcount == MPI_UNDEFINED || outcount == 0) {
    return;
  }

  // Populate the output indices and update the completed C++ objects
  completed_indices.reserve(outcount);
  for (int i = 0; i < outcount; ++i) {
    int idx = raw_indices[i];
    completed_indices.push_back(idx);

    // Update our C++ objects so they know they are finished (now NULL)
    requests[idx].update(MPI_Request_c2f(c_reqs[idx]));
  }
}

} // namespace ttnte::parallel
