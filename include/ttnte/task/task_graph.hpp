#pragma once

#include "ttnte/task/task.hpp"
#include "ttnte/utils/label.hpp"
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ttnte::task {

/// @brief The TaskGraph is in charge of adding Tasks and new dependencies. This
/// class handles the directed asyclic graph (DAG) of functions to execute.
class TaskGraph {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<TaskGraph>;
  using TaskDeque = std::deque<Task>;

  /// @brief Index form of the DAG for push-based execution: successors and
  /// dependency counts of every task (tasks indexed in creation order).
  struct PushPlan {
    /// Tasks that become ready when task i completes.
    std::vector<std::vector<int64_t>> successors;
    /// Number of dependencies of task i.
    std::vector<int64_t> num_dependencies;
    /// Tasks without dependencies.
    std::vector<int64_t> roots;
    /// Whether task i is a network task (run by the calling thread).
    std::vector<uint8_t> is_network;
    /// Whether every task is CPU_ASYNC or NETWORK_* (what PUSH supports).
    bool supported = true;
    /// Number of tasks and edges the plan was built from (staleness check).
    size_t num_tasks = 0;
    size_t num_edges = 0;
  };

private:
  // =================================================================
  // Private data
  /// Label of the task graph.
  Label label_;
  /// Deque of tasks.
  TaskDeque tasks_;

  /// Cached push-based execution plan (see push_plan()).
  PushPlan push_plan_;

  /// Mutex for a thread lock.
  std::mutex mutex;

public:
  // =================================================================
  // Public constructor
  /// @brief Constructor of the task graph.
  /// @param Label of the task graph.
  TaskGraph(std::optional<std::string> label = std::nullopt)
    : label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}

  // =================================================================
  // Public methods
  /// @brief Create a new task in the DAG.
  /// @param target Target device for the task to execute on.
  /// @param label Label of the task.
  Task* create_task(std::optional<DeviceTarget> target = std::nullopt,
    std::optional<std::string> label = std::nullopt);
  /// @brief Add a dependency to a task.
  /// @param dependent The task to add the dependency to.
  /// @param dependency The task to add as a dependency.
  void add_dependency(Task* dependent, Task* dependency);
  /// @brief Clear the task graph.
  void clear();
  /// @brief Index form of the DAG, rebuilt whenever tasks or dependencies
  /// were added since it was last built (dependencies may be added through
  /// Task::add_dependency(), so staleness is detected from the edge count).
  /// @return The push-based execution plan.
  const PushPlan& push_plan();
  /// @brief Reset all tasks to WAITING so the graph can be re-executed without
  /// rebuilding.
  void reset()
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& task : tasks_)
      task.reset();
  }

  // =================================================================
  // Public getters / setters
  /// @return The label of the task graph.
  const Label& get_label() const noexcept { return label_; }
  /// @return The deque of tasks.
  std::deque<Task>& get_tasks() { return tasks_; }
  /// @return The number of tasks in the graph.
  size_t size() const noexcept { return tasks_.size(); }
};

} // namespace ttnte::task
