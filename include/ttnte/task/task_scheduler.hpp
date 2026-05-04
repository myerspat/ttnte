#pragma once

#include "ttnte/parallel/thread_pool.hpp"
#include "ttnte/task/task_graph.hpp"
#include "ttnte/utils/label.hpp"
#include <mutex>

namespace ttnte::task {

/// @brief This class executes the graph held by the TaskGraph.
class TaskScheduler {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<TaskScheduler>;

private:
  // =================================================================
  // Private data
  /// Label of the scheduler.
  Label label_;
  /// The thread pool of the scheduler.
  parallel::ThreadPool thread_pool_;

  /// The mutex for stopping race conditions.
  std::mutex mutex;

public:
  // =================================================================
  // Public constructor
  /// @brief Constructor for the scheduler.
  /// @param num_threads The number of threads for the thread pool.
  /// @param label The label of the scheduler.
  TaskScheduler(
    size_t num_threads = 4, std::optional<std::string> label = std::nullopt)
    : thread_pool_(num_threads),
      label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}

  // =================================================================
  // Public methods
  /// @brief Execute a task graph.
  /// @param The TaskGraph to execute.
  void execute(TaskGraph& graph);

  // =================================================================
  // Public getters / setters
  /// @return The label of the scheduler.
  const Label& get_label() const noexcept { return label_; }
  /// @return The thread pool of the scheduler.
  const parallel::ThreadPool& get_thread_pool() const noexcept
  {
    return thread_pool_;
  }
};

} // namespace ttnte::task
