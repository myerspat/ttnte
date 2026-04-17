#pragma once

#include "ttnte/task/task.hpp"
#include "ttnte/utils/label.hpp"
#include <deque>
#include <mutex>
#include <optional>
#include <string>

namespace ttnte::task {

/// @brief The TaskGraph is in charge of adding Tasks and new dependencies.
class TaskGraph {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<TaskGraph>;
  using TaskDeque = std::deque<Task>;

private:
  // =================================================================
  // Private data
  Label label_;
  TaskDeque tasks_;

  // Thread lock
  std::mutex mutex;

public:
  // =================================================================
  // Public constructor
  TaskGraph(std::optional<std::string> label = std::nullopt)
    : label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}

  // =================================================================
  // Public methods
  Task* create_task(std::optional<DeviceTarget> target,
    std::optional<std::string> label = std::nullopt);
  void add_dependency(Task* dependent, Task* dependency);
  void clear();

  // =================================================================
  // Public getters / setters
  std::deque<Task>& get_tasks() { return tasks_; }
  size_t size() const noexcept { return tasks_.size(); }
};

} // namespace ttnte::task
