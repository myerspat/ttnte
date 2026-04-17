#include "ttnte/task/task_scheduler.hpp"

namespace ttnte::task {

void TaskScheduler::execute(TaskGraph& graph)
{
  // Lock class from multiple threads trying to execute the same graph
  std::lock_guard<std::mutex> lock(mutex);

  int tasks_completed = 0;
  const int total_tasks = graph.size();

  while (tasks_completed < total_tasks) {
    bool made_progress = false;

    for (auto& task : graph.get_tasks()) {
      TaskStatus current_status = task.get_status();

      // Check dependencies
      if (current_status == TaskStatus::WAITING && task.check_dependencies()) {
        current_status = TaskStatus::READY;
        task.update_status(current_status);
        made_progress = true;
      }

      // Dispatch READY tasks
      if (current_status == TaskStatus::READY) {

        // Transition out of READY on the main thread before dispatching
        current_status = TaskStatus::RUNNING;
        task.update_status(current_status);
        made_progress = true;

        DeviceTarget target = task.get_target();

        if (target == DeviceTarget::CPU_SYNC ||
            target == DeviceTarget::GPU_SYNC ||
            target == DeviceTarget::NETWORK_SYNC) {

          // Execute blocking payload directly on the main scheduler thread
          task.execute();
          assert(task.get_status() == TaskStatus::COMPLETED);

        } else { // All _ASYNC targets

          // Hand off the initial dispatch to a background worker
          thread_pool_.push_task([&task]() {
            // The payload performs the work or hardware dispatch, and
            // inherently updates its own atomic status to POLLING or COMPLETED
            task.execute();
          });
        }
      }

      // Fetch status again in case an async thread just finished its dispatch
      current_status = task.get_status();

      // Poll hardware for asynchronous tasks
      if (current_status == TaskStatus::POLLING) {
        // The payload now acts as a non-blocking check
        task.execute();

        // Fetch status one last time for the final count check
        current_status = task.get_status();
      }

      // Mark completed tasks
      if (current_status == TaskStatus::COMPLETED && !task.is_counted()) {
        tasks_completed++;
        task.count();
        made_progress = true;
      }
    }

    // If a full pass of the DAG resulted in no dependencies met, no dispatches,
    // and no completed polls, yield the CPU time slice to the OS.
    if (!made_progress) {
      std::this_thread::yield();
    }
  }
}

} // namespace ttnte::task
