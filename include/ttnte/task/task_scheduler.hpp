#pragma once

#include "ttnte/parallel/thread_pool.hpp"
#include "ttnte/task/task_graph.hpp"
#include "ttnte/utils/label.hpp"
#include <mutex>

namespace ttnte::task {

class TaskScheduler {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<TaskScheduler>;

private:
  // =================================================================
  // Private data
  Label label_;
  parallel::ThreadPool thread_pool_;

  // Thread lock
  std::mutex mutex;

public:
  // =================================================================
  // Public constructor
  TaskScheduler(
    size_t num_threads = 4, std::optional<std::string> label = std::nullopt)
    : thread_pool_(num_threads),
      label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}

  // =================================================================
  // Public methods
  void execute(TaskGraph& graph);

  // =================================================================
  // Public getters / setters
  const Label& get_label() const noexcept { return label_; }
  const parallel::ThreadPool& get_thread_pool() const noexcept
  {
    return thread_pool_;
  }
};

} // namespace ttnte::task
