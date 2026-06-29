#pragma once

#include "ttnte/utils/label.hpp"
#include "ttnte/utils/trigger_global_crach.hpp"
#include <atomic>
#include <c10/core/Device.h>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ttnte::task {

/// @brief Status enum for tasks.
enum class TaskStatus : uint8_t {
  WAITING,  // Waiting for the task to be ready to run
  READY,    // Task is ready to run
  RUNNING,  // Task is running
  POLLING,  // MPI_Isend or MPI_Irecv needs to be checked using MPI_Test
  COMPLETED // This task is complete
};

/// @brief The target device that the task will run on.
enum class DeviceTarget : uint8_t {
  CPU_SYNC,      // CPU only workload (blocking)
  CPU_ASYNC,     // CPU only workload (non-blocking)
  GPU_SYNC,      // GPU workload (blocking)
  GPU_ASYNC,     // GPU workload (non-blocking)
  NETWORK_SYNC,  // Network workload using MPI (blocking)
  NETWORK_ASYNC, // Network workload using MPI (non-blocking)
};

/// @brief The task class for a task-based solver using a directed asyclic
/// graph.
class Task {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<Task>;
  using Ptr = std::unique_ptr<Task>;

private:
  // =================================================================
  // Private data
  /// Label of the task.
  Label label_;
  /// Primary device.
  DeviceTarget target_;
  /// Status of the task.
  std::atomic<TaskStatus> status_ {TaskStatus::WAITING};
  /// Vector of dependent tasks.
  std::vector<Task*> dependencies_;
  /// Function to run when TaskStatus::READY.
  std::function<TaskStatus()> payload_;
  /// Whether the task's completion has been counted.
  bool is_counted_ = false;
  /// Whether the task is currently executing
  std::atomic_flag is_executing_ = ATOMIC_FLAG_INIT;

public:
  // =================================================================
  // Public constructors
  Task(std::optional<DeviceTarget> target = std::nullopt,
    std::optional<std::string> label = std::nullopt)
    : label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {
    if (target.has_value()) {
      target_ = target.value();
    }
  }

  // =================================================================
  // Public methods
  /// @brief Add a dependency that this task must wait for. Once all
  /// dependencies are TaskStatus::COMPLETED then this task is
  /// TaskStatus::READY.
  /// @param dependency A pointer to the task that this task depends on.
  void add_dependency(Task* dependency) { dependencies_.push_back(dependency); }

  /// @brief Check if the dependencies are have completed.
  /// @return Whether all dependencies are done.
  bool check_dependencies() const
  {
    for (Task* dependency : dependencies_) {
      if (dependency->get_status() != TaskStatus::COMPLETED) {
        return false;
      }
    }
    return true;
  }

  /// @brief Update the TaskStatus of this task.
  /// @param new_status The new status of this task.
  void update_status(TaskStatus new_status)
  {
    status_.store(new_status, std::memory_order_release);
  }

  /// @brief Execute the function that this task wraps.
  void execute()
  {
    if (is_executing_.test_and_set(std::memory_order_acquire)) {
      return;
    }

    try {
      update_status(payload_());

    } catch (const c10::Error& e) {
      ttnte::utils::trigger_global_crash("LibTorch error on task " +
                                         label_.to_string() + ": " +
                                         std::string(e.what()));

    } catch (const std::exception& e) {
      ttnte::utils::trigger_global_crash("C++ exception in task " +
                                         label_.to_string() + ": " +
                                         std::string(e.what()));

    } catch (...) {
      ttnte::utils::trigger_global_crash("Unknown exception in DAG task");
    }

    // Release the executing flag
    is_executing_.clear(std::memory_order_release);
  }

  /// @brief Check if this status has been counted as completed.
  bool is_counted() const noexcept { return is_counted_; }

  /// @brief Change this task to counted.
  void count()
  {
    assert(get_status() == TaskStatus::COMPLETED);
    is_counted_ = true;
  }

  /// @Brief Reset this task to waiting.
  void reset()
  {
    update_status(TaskStatus::WAITING);
    is_counted_ = false;
  }

  // =================================================================
  // Public Getters / Setters
  /// @return The label of this class.
  const Label& get_label() const noexcept { return label_; }
  /// @return The target device for this task.
  const DeviceTarget& get_target() const noexcept { return target_; }
  /// @return The status of this task.
  TaskStatus get_status() const noexcept
  {
    return status_.load(std::memory_order_acquire);
  }

  /// @param The new target device for this task.
  void set_target(DeviceTarget new_target) { target_ = new_target; }
  /// @param The payload function that this task will execute.
  void set_payload(std::function<TaskStatus()> payload) { payload_ = payload; }
};

} // namespace ttnte::task
