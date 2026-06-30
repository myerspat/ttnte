#pragma once

#include "ttnte/linalg/neighbor_coupling.hpp"
#include "ttnte/linalg/source.hpp"
#include "ttnte/linalg/tt_config.hpp"
#include "ttnte/task/task.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::task::cpu {

/// @brief Configure a task to execute `solver->solve(system)` on CPU.
/// @param task The task to add the payload to.
/// @param system The system passed to the `solver`.
/// @param solver The solver with a `solve()` method.
template<typename DataType, typename SolverType>
Task& configure_solve_task(Task& task, const std::shared_ptr<DataType>& system,
  const std::shared_ptr<SolverType>& solver)
{
  if (task.get_target() == DeviceTarget::CPU_ASYNC ||
      task.get_target() == DeviceTarget::CPU_SYNC) {
    // Asynchronous solve task
    task.set_payload([system, solver]() mutable -> TaskStatus {
      // Dispatch solve kernel: this thread will not get passed the solve method
      // until it's done
      solver->solve(system);
      return TaskStatus::COMPLETED;
    });

  } else {
    throw utils::runtime_error("ttnte::task::cpu::configure_solve_task\n",
      "The target device of the `task` must be either\n"
      "`ttnte::task::DeviceTarget::CPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::CPU_ASYNC`");
  }

  return task;
}

/// @brief Configure the full eigenvalue compute step as a single task:
///   1. Set source->state_ = eigval * F*phi so the solver reads the scaled RHS.
///   2. Solve — AMEn accumulates boundary contributions and writes phi_new.
///   3. Clear source->state_ to free the scaled source before recomputing.
///   4. Recompute F*phi_new = mv(fission_op, phi_new) and cache total_source_
///      for the driver's k-update.
/// @param task The task to configure.
/// @param system The linear system to solve.
/// @param eigen_src The eigenvalue source owning the fission operator.
/// @param solver The local solver.
/// @param eps TT rounding tolerance for the update step.
/// @param max_rank Maximum TT rank for the update step.
template<typename DataType, typename SolverType>
Task& configure_eigensolve_task(Task& task,
  const std::shared_ptr<DataType>& system, const linalg::EigenSource::Ptr& src,
  const std::shared_ptr<SolverType>& solver,
  const std::shared_ptr<linalg::TTConfig>& config)
{
  task.set_payload([system, src, solver, config]() mutable -> TaskStatus {
    // Compute the updated eigen source
    src->scale();

    // Solve the linear system
    solver->solve(system);

    // Clear the old eigen source
    src->update(system->get_state(), config->eps, config->max_rank);
    return TaskStatus::COMPLETED;
  });
  return task;
}

/// @brief Configure the apply task for apply the boundary operator of a coupled
/// face to the received buffer.
/// @param task The task to configure.
/// @param coupling The coupled face with the filled receive buffer.
/// @param config The settings for low-rank tensor networks.
/// @return The configured task.
inline Task& configure_apply_task(Task& task,
  linalg::NeighborCoupling* coupling,
  const std::shared_ptr<linalg::TTConfig>& config)
{
  task.set_payload([coupling, config]() mutable -> TaskStatus {
    if (!coupling->recv_buffer.defined()) {
      return TaskStatus::COMPLETED;
    }

    // Apply the mapping to the current receive buffer
    coupling->set_recv_buffer(
      std::move(coupling->recv_buffer), true, config->eps, config->max_rank);

    // Apply the boundary operator and round
    coupling->recv_buffer =
      linalg::mv(coupling->boundary_op, coupling->recv_buffer);
    coupling->recv_buffer.round_(config->eps, config->max_rank);

    return TaskStatus::COMPLETED;
  });

  return task;
}

/// @brief Configure a task to isolate the boundaries of a state and fill the
/// send buffer for the coupled face.
/// @param task The task to configure.
/// @param system The solved linear system.
/// @param coupling The coupled face with the filled receive buffer.
/// @param config The settings for low-rank tensor networks.
/// @return The configured task.
template<typename DataType>
inline Task& configure_narrow_task(Task& task,
  const std::shared_ptr<DataType>& system, linalg::NeighborCoupling* coupling,
  const std::shared_ptr<linalg::TTConfig>& config)
{
  task.set_payload([system, coupling, config]() mutable -> TaskStatus {
    // Use narrow() (non-mutating) to extract the boundary face. A copy
    // of State is a shallow shared-pointer copy; narrow_() on that copy
    // would mutate the system's live state, corrupting future iterations.
    const auto& state = system->get_state();
    const size_t boundary_dim = static_cast<size_t>(state.ndimension()) -
                                coupling->connection.mapping.flip.size() - 2 +
                                coupling->dim;
    auto face = state.narrow(boundary_dim, coupling->is_upper ? -1 : 0, 1);
    face.round_(config->eps, config->max_rank);
    coupling->set_send_buffer(std::move(face));

    return TaskStatus::COMPLETED;
  });

  return task;
}

/// @brief Pack the outgoing boundary data into a single buffer for MPI
/// communication.
/// @param task The task to configure.
/// @param coupling The coupled face with the filled receive buffer.
/// @param send_buffer The outgoing buffer to fill.
/// @return The configured task.
inline Task& configure_pack_task(Task& task, linalg::NeighborCoupling* coupling,
  const std::shared_ptr<torch::Tensor>& send_buffer)
{
  task.set_payload([coupling, send_buffer]() -> TaskStatus {
    *send_buffer = coupling->send_buffer.pack();
    coupling->set_send_buffer(linalg::State());
    return TaskStatus::COMPLETED;
  });
  return task;
}

/// @brief Configure the unpack tasks for unpacking a boundary buffer received
/// over MPI.
/// @param task The task to configure.
/// @param coupling The coupled face with the filled receive buffer.
/// @param recv_buffer The incoming buffer filled by MPI.
/// @return The configured task.
inline Task& configure_unpack_task(Task& task,
  linalg::NeighborCoupling* coupling,
  const std::shared_ptr<torch::Tensor>& recv_buffer)
{
  task.set_payload([coupling, recv_buffer]() -> TaskStatus {
    coupling->set_recv_buffer(linalg::State::unpack(*recv_buffer), false);
    *recv_buffer = torch::Tensor();
    return TaskStatus::COMPLETED;
  });
  return task;
}

/// @brief Configure the boundary exchange for local boundary transfers (non-MPI
/// or same rank).
/// @param task The task to configure.
/// @param target_coupling The coupled face receiving the boundary data.
/// @param source_coupling The coupled face sharing its boundary data.
/// @return The configured task.
inline Task& configure_within_rank_send(Task& task,
  linalg::NeighborCoupling* target_coupling,
  linalg::NeighborCoupling* source_coupling)
{
  task.set_payload([target_coupling, source_coupling]() -> TaskStatus {
    target_coupling->set_recv_buffer(
      std::move(source_coupling->send_buffer), false);
    source_coupling->set_send_buffer(linalg::State());
    return TaskStatus::COMPLETED;
  });
  return task;
}

} // namespace ttnte::task::cpu
