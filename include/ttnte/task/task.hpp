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

enum class TaskStatus : uint8_t {
  WAITING,  // Waiting for the task to be ready to run
  READY,    // Task is ready to run
  RUNNING,  // Task is running
  POLLING,  // MPI_Isend or MPI_Irecv needs to be checked using MPI_Test
  COMPLETED // This task is complete
};

enum class DeviceTarget : uint8_t {
  CPU_SYNC,      // CPU only workload (blocking)
  CPU_ASYNC,     // CPU only workload (non-blocking)
  GPU_SYNC,      // GPU workload (blocking)
  GPU_ASYNC,     // GPU workload (non-blocking)
  NETWORK_SYNC,  // Network workload using MPI (blocking)
  NETWORK_ASYNC, // Network workload using MPI (non-blocking)
};

class Task {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<Task>;
  using Ptr = std::unique_ptr<Task>;

private:
  // =================================================================
  // Private data
  /// Label of the task
  Label label_;
  /// Primary device
  DeviceTarget target_;
  /// Status of the task
  std::atomic<TaskStatus> status_ {TaskStatus::WAITING};
  /// Vector of dependent tasks
  std::vector<Task*> dependencies_;
  /// Function to run when READY
  std::function<TaskStatus()> payload_;
  /// Whether the task's completion has been counted
  bool is_counted_ = false;

public:
  // =================================================================
  // Public constructors
  Task(std::optional<DeviceTarget> target,
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
  void add_dependency(Task* dependency) { dependencies_.push_back(dependency); }

  bool check_dependencies() const
  {
    for (Task* dependency : dependencies_) {
      if (dependency->status_.load() != TaskStatus::COMPLETED) {
        return false;
      }
    }
    return true;
  }

  void update_status(TaskStatus new_status)
  {
    status_.store(new_status, std::memory_order_release);
  }

  void execute()
  {
    try {
      status_.store(payload_());

    } catch (const c10::Error& e) {
      ttnte::utils::trigger_global_crash(
        "LibTorch Error: " + std::string(e.what()));

    } catch (const std::exception& e) {
      ttnte::utils::trigger_global_crash(
        "C++ exception: " + std::string(e.what()));

    } catch (...) {
      ttnte::utils::trigger_global_crash("Unknown exception in DAG task");
    }
  }

  bool is_counted() const noexcept { return is_counted_; }

  void count()
  {
    assert(get_status() == TaskStatus::COMPLETED);
    is_counted_ = true;
  }

  void reset()
  {
    status_.store(TaskStatus::WAITING);
    is_counted_ = false;
  }

  // =================================================================
  // Public Getters / Setters
  const Label& get_label() const noexcept { return label_; }
  const DeviceTarget& get_target() const noexcept { return target_; }
  TaskStatus get_status() const noexcept { return status_.load(); }

  void set_target(DeviceTarget new_target) { target_ = new_target; }
  void set_payload(std::function<TaskStatus()> payload) { payload_ = payload; }
};

} // namespace ttnte::task
