#include "ttnte/parallel/thread_pool.hpp"

namespace ttnte::parallel {

// =================================================================
// Private constructors
ThreadPool::ThreadPool(size_t num_threads) : stop_(false)
{
  {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;

          {
            // Lock the queue to safely pull a task
            std::unique_lock<std::mutex> lock(this->queue_mutex_);

            // The thread sleeps here until notified OR the pool is stopped
            this->cv_.wait(
              lock, [this] { return this->stop_ || !this->tasks_.empty(); });

            if (this->stop_ && this->tasks_.empty()) {
              return; // Exit the thread cleanly
            }

            // Grab the task and remove it from the queue
            task = std::move(this->tasks_.front());
            this->tasks_.pop();
          }

          // Execute the task outside the lock so other threads can pull tasks!
          task();
        }
      });
    }
  }
}

ThreadPool::~ThreadPool()
{
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }

  // Wake up all workers so they check the stop_ flag and exit
  cv_.notify_all();

  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

} // namespace ttnte::parallel
