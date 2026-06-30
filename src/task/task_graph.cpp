#include "ttnte/task/task_graph.hpp"

namespace ttnte::task {

// =================================================================
// Public methods
Task* TaskGraph::create_task(
  std::optional<DeviceTarget> target, std::optional<std::string> label)
{
  // Lock class from multiple threads calling
  std::lock_guard<std::mutex> lock(mutex);
  tasks_.emplace_back(target, label);
  return &tasks_.back();
}

void TaskGraph::add_dependency(Task* dependent, Task* dependency)
{
  // Lock class from multiple threads calling
  std::lock_guard<std::mutex> lock(mutex);
  dependent->add_dependency(dependency);
}

void TaskGraph::clear()
{
  // Lock class from multiple threads calling
  std::lock_guard<std::mutex> lock(mutex);
  tasks_.clear();
}

} // namespace ttnte::task
