#include "ttnte/task/task_graph.hpp"
#include "ttnte/utils/exception.hpp"
#include <unordered_map>

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

const TaskGraph::PushPlan& TaskGraph::push_plan()
{
  // Lock class from multiple threads calling
  std::lock_guard<std::mutex> lock(mutex);

  size_t num_edges = 0;
  for (const auto& task : tasks_)
    num_edges += task.get_dependencies().size();
  if (push_plan_.num_tasks == tasks_.size() &&
      push_plan_.num_edges == num_edges && !tasks_.empty())
    return push_plan_;

  const size_t n = tasks_.size();
  std::unordered_map<const Task*, int64_t> index;
  index.reserve(n);
  for (size_t i = 0; i < n; i++)
    index.emplace(&tasks_[i], static_cast<int64_t>(i));

  PushPlan plan;
  plan.successors.assign(n, {});
  plan.num_dependencies.assign(n, 0);
  plan.is_network.assign(n, 0);
  for (size_t i = 0; i < n; i++) {
    const Task& task = tasks_[i];
    const auto target = task.get_target();
    plan.is_network[i] = target == DeviceTarget::NETWORK_ASYNC ||
                         target == DeviceTarget::NETWORK_SYNC;
    plan.supported = plan.supported &&
                     (target == DeviceTarget::CPU_ASYNC || plan.is_network[i]);
    for (const Task* dependency : task.get_dependencies()) {
      auto it = index.find(dependency);
      if (it == index.end()) {
        throw utils::runtime_error("ttnte::task::TaskGraph::push_plan",
          "Task " + task.get_label().to_string() +
            " depends on a task outside this graph");
      }
      plan.successors[it->second].push_back(static_cast<int64_t>(i));
      plan.num_dependencies[i]++;
    }
    if (plan.num_dependencies[i] == 0)
      plan.roots.push_back(static_cast<int64_t>(i));
  }
  plan.num_tasks = n;
  plan.num_edges = num_edges;
  push_plan_ = std::move(plan);
  return push_plan_;
}

void TaskGraph::clear()
{
  // Lock class from multiple threads calling
  std::lock_guard<std::mutex> lock(mutex);
  tasks_.clear();
  push_plan_ = PushPlan();
}

} // namespace ttnte::task
