#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ttnte::parallel {

class ThreadPool {
private:
  // =================================================================
  // Private data
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex queue_mutex_;
  std::condition_variable cv_;
  bool stop_;

public:
  // =================================================================
  // Public constructors
  explicit ThreadPool(size_t num_threads = 4);
  ~ThreadPool();

  // Prevent copying which would wreck the thread management
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // =================================================================
  // Public methods
  template<class FuncType>
  void push_task(FuncType&& f)
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("Cannot push tasks into a stopped ThreadPool");
      }
      tasks_.emplace(std::forward<FuncType>(f));
    }
    // Wake up exactly one sleeping worker to handle this new task
    cv_.notify_one();
  }
};

} // namespace ttnte::parallel
