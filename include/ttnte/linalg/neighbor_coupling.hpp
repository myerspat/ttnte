#pragma once

#include "ttnte/linalg/operator.hpp"
#include "ttnte/linalg/state.hpp"
#include "ttnte/mesh/mesh_block_boundary.hpp"
#include <cstddef>
#include <limits>

namespace ttnte::linalg {

/// @brief Coupling from a neighboring patch at an INTERNAL boundary face.
///
/// Produced by the assembler for each INTERNAL-type boundary and stored on
/// the LinearSystem. During a block-Jacobi iteration the DAG scheduler:
///   1. narrows the neighbor's solution TT along dim at the boundary face,
///   2. applies the boundary mapping (flip / permute) via set_recv_buffer,
///   3. multiplies by boundary_op to form the coupling contribution, and
///   4. accumulates that contribution into the RHS before the local AMEn solve.
struct NeighborCoupling {
  // =================================================================
  // Data
  /// The face ID on this rank.
  size_t fid;
  /// Neighbor GID, face index on the neighbor, MPI rank, and axis mapping.
  mesh::NeighborInfo connection;
  /// Inflow operator assembled with n_mode=1 at the boundary spatial core.
  Operator boundary_op;
  /// Physical dimension index of the INTERNAL face (0, 1, ..., NumDim-1).
  size_t dim;
  /// True if the face is the upper face along dim, false if lower.
  bool is_upper;
  /// Receive buffer written by the narrow/unpack task, read by the apply task,
  /// and cleared by the solve task after accumulation.
  State recv_buffer = State();

  State send_buffer = State();

  /// Squared L2 norm of (face_new − face_old) / 2, written by AMEnSolver::solve
  /// each Jacobi sweep by narrowing the new and previous solutions at this
  /// boundary. The driver sums sq_diff and sq_prev across all couplings and
  /// patches to form the global relative boundary L2 error:
  /// sqrt(Σ sq_diff) / sqrt(Σ sq_prev). Both are 0 on the first sweep (no
  /// previous solution yet), causing the driver to treat it as not converged.
  double sq_diff = 0.0;
  /// Squared L2 norm of face_old / 2 (denominator contribution).
  double sq_prev = 0.0;

  // =================================================================
  // Methods
  /// @brief Apply the boundary mapping to face and store it in recv_buffer.
  ///
  /// First permutes the physical dimensions of the face state so that spatial
  /// axes arrive in this patch's coordinate ordering (via
  /// connection.mapping.perm), then flips any perpendicular spatial dimensions
  /// whose index ordering runs opposite to the neighbor's (via
  /// connection.mapping.flip). State::permute_ and State::flip_ dispatch to the
  /// underlying engine so this works for any State type (TT or Dense).
  ///
  /// @param face          Narrowed face state from the neighbor's solution.
  /// @param apply_mapping Whether to apply the mapping. Pass false to skip for
  /// testing.
  void set_recv_buffer(State face, bool apply_mapping = true, double eps = 0,
    int64_t max_rank = std::numeric_limits<int64_t>::max());

  void set_send_buffer(State face) { send_buffer = std::move(face); }
};

} // namespace ttnte::linalg
