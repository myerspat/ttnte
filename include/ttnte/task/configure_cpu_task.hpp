#pragma once

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
      "`ttnte::task::DeviceTarget::GPU_SYNC` or\n"
      "`ttnte::task::DeviceTarget::GPU_ASYNC`");
  }

  return task;
}

} // namespace ttnte::task::cpu
