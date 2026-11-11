#pragma once

#include <exception>
#include <mutex>

namespace ttnte::bindings {

/// @brief The first exception raised by a Python task payload since the last
/// TaskScheduler.execute(), which rethrows it. A payload's exception would
/// otherwise reach Task::execute(), which aborts the process (by design, for
/// C++ tasks under MPI); Python prototypes get an ordinary exception.
struct PayloadError {
  std::mutex mutex;
  std::exception_ptr error;

  /// @return The process-wide slot.
  static PayloadError& instance()
  {
    static PayloadError slot;
    return slot;
  }

  /// @brief Keep `e` if no earlier payload failed.
  void record(std::exception_ptr e)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!error)
      error = std::move(e);
  }

  /// @brief Rethrow (and clear) the recorded exception, if any.
  void rethrow()
  {
    std::exception_ptr e;
    {
      std::lock_guard<std::mutex> lock(mutex);
      std::swap(e, error);
    }
    if (e)
      std::rethrow_exception(e);
  }
};

} // namespace ttnte::bindings
