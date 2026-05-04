#pragma once

#include "ttnte/task/task.hpp"
#include "ttnte/utils/label.hpp"
#include <deque>
#include <mutex>
#include <optional>
#include <string>

namespace ttnte::task {

/// @brief The TaskGraph is in charge of adding Tasks and new dependencies. This
/// class handles the directed asyclic graph (DAG) of functions to execute.
class TaskGraph {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<TaskGraph>;
  using TaskDeque = std::deque<Task>;

private:
  // =================================================================
  // Private data
  /// Label of the task graph.
  Label label_;
  /// Deque of tasks.
  TaskDeque tasks_;

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
  Task* create_task(std::optional<DeviceTarget> target,
    std::optional<std::string> label = std::nullopt);
  /// @brief Add a dependency to a task.
  /// @param dependent The task to add the dependency to.
  /// @param dependency The task to add as a dependency.
  void add_dependency(Task* dependent, Task* dependency);
  /// @brief Clear the task graph.
  void clear();

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
