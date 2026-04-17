#pragma once

#include "ttnte/parallel/communicator.hpp"
#include "ttnte/utils/exception.hpp"
#include <c10/util/SmallVector.h>
#include <cstddef>

namespace ttnte::parallel {

class BoundaryCommunicator {
public:
  // =================================================================
  // Public types
  using BoundaryComms = c10::SmallVector<Communicator, 6>;

private:
  // =================================================================
  // Private data
  BoundaryComms boundary_comms_;

  // =================================================================
  // Private methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::parallel::BoundaryCommunicator::" + func_name;
  }

public:
  // Add these to be explicit
  BoundaryCommunicator(const BoundaryCommunicator&) = delete;
  BoundaryCommunicator& operator=(const BoundaryCommunicator&) = delete;

  // =================================================================
  // Public constructors
  BoundaryCommunicator(const Communicator& world_comm, int64_t num_boundaries)
  {
    if (!world_comm.is_world_comm()) {
      throw utils::runtime_error(error_context("BoundaryCommunicator"),
        "The Communicator given to the BoundaryCommunicator is not the "
        "MPI_WORLD_COMM");
    }

    boundary_comms_.reserve(num_boundaries);
    for (size_t i = 0; i < num_boundaries; i++) {
      boundary_comms_.push_back(world_comm.duplicate());
    }
  }

  // =================================================================
  // Public methods
  static int32_t generate_tag(
    int32_t sender_id, int sender_level, int32_t recv_id, int recv_level)
  {
    if (sender_level < recv_level)
      return recv_id;
    return sender_id;
  }

  // =================================================================
  // Public setters / getters
  const BoundaryComms& get_comms() const noexcept { return boundary_comms_; }
  const Communicator& get_comm(int32_t fid) const
  {
    if (fid < 0 || fid >= boundary_comms_.size()) {
      throw utils::runtime_error(error_context("get_comm"),
        "`fid = " + std::to_string(fid) + "` is out of range");
    }

    return boundary_comms_[fid];
  }
};

} // namespace ttnte::parallel
