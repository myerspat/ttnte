#include "ttnte/task/task_scheduler.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/utils/exception.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <torch/cuda.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAFunctions.h>

namespace {
void pin_cuda_device(torch::DeviceIndex device_idx)
{
  c10::cuda::set_device(device_idx);
}
} // namespace

#else

namespace {
void pin_cuda_device(torch::DeviceIndex device_idx) {}
} // namespace

#endif

namespace {

/// @brief Build the per-worker-thread startup routine: pin the CUDA device
/// (matching what ThreadPool used to do internally) and permanently claim
/// one stream from `stream_pool` for this thread (see
/// StreamPool::claim_for_this_thread() -- the reason every worker needs its
/// own, never-shared stream; a no-op on a CPU-only build/run, since the pool
/// is then empty).
std::function<void()> build_worker_init(
  const ttnte::parallel::StreamPool::Ptr& stream_pool)
{
  auto device_idx =
    ttnte::parallel::ParallelContext::instance().device().index();

  return [device_idx, stream_pool]() {
    if (torch::cuda::is_available()) {
      pin_cuda_device(device_idx);
    }
    stream_pool->claim_for_this_thread();
  };
}

} // namespace

namespace {

/// @brief State of one ExecutionMode::PUSH execution. Shared (via
/// shared_from_this) by every queued closure, so a worker still unwinding
/// after the last completion never touches freed memory.
struct PushRun : std::enable_shared_from_this<PushRun> {
  ttnte::parallel::ThreadPool* pool = nullptr;
  ttnte::task::TaskGraph::TaskDeque* tasks = nullptr;
  const ttnte::task::TaskGraph::PushPlan* plan = nullptr;
  /// Unfinished dependencies of every task.
  std::unique_ptr<std::atomic<int64_t>[]> pending;
  /// Unfinished tasks.
  std::atomic<size_t> remaining {0};
  /// Whether a CPU payload returned something other than COMPLETED.
  std::atomic<bool> incomplete {false};
  /// Guards network_ready and done; cv wakes the calling (main) thread.
  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;
  /// Released network tasks waiting for the main thread.
  std::vector<int64_t> network_ready;

  void enqueue_cpu(int64_t i)
  {
    auto self = shared_from_this();
    pool->push_task([self, i]() { self->run_cpu(i); });
  }

  void enqueue_network(int64_t i)
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      network_ready.push_back(i);
    }
    cv.notify_one();
  }

  /// @brief Completion bookkeeping of task i: release its successors
  /// (network ones to the main thread, CPU ones to the pool).
  /// @param keep_one Return one released CPU successor instead of queueing
  /// it, for the calling worker to run next (it consumes what i produced).
  /// @return That successor, or -1.
  int64_t finish(int64_t i, bool keep_one)
  {
    auto& task = (*tasks)[i];
    if (task.get_status() == ttnte::task::TaskStatus::COMPLETED)
      task.count();
    else
      incomplete.store(true, std::memory_order_relaxed);

    int64_t next = -1;
    for (int64_t c : plan->successors[i]) {
      if (pending[c].fetch_sub(1, std::memory_order_acq_rel) != 1)
        continue;
      if (plan->is_network[c])
        enqueue_network(c);
      else if (keep_one && next < 0)
        next = c;
      else
        enqueue_cpu(c);
    }
    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lock(mutex);
      done = true;
      cv.notify_one();
    }
    return next;
  }

  /// @brief Worker: run CPU task i and keep running released successors.
  void run_cpu(int64_t i)
  {
    while (i >= 0) {
      auto& task = (*tasks)[i];
      task.update_status(ttnte::task::TaskStatus::RUNNING);
      task.execute();
      i = finish(i, true);
    }
  }

  /// @brief Calling thread: start and poll network tasks (MPI stays on this
  /// one thread) until every task has finished; sleeps when no request is
  /// in flight.
  void drive()
  {
    using ttnte::task::TaskStatus;
    std::vector<int64_t> in_flight, ready;
    std::unique_lock<std::mutex> lock(mutex);
    while (!done) {
      ready.swap(network_ready);
      if (ready.empty() && in_flight.empty()) {
        cv.wait(lock, [this] { return done || !network_ready.empty(); });
        continue;
      }
      lock.unlock();

      bool progressed = false;
      for (int64_t i : ready) {
        auto& task = (*tasks)[i];
        task.update_status(TaskStatus::RUNNING);
        task.execute();
        if (task.get_status() == TaskStatus::POLLING) {
          in_flight.push_back(i);
        } else {
          finish(i, false);
          progressed = true;
        }
      }
      ready.clear();
      for (size_t k = 0; k < in_flight.size();) {
        auto& task = (*tasks)[in_flight[k]];
        task.execute();
        if (task.get_status() == TaskStatus::POLLING) {
          k++;
          continue;
        }
        finish(in_flight[k], false);
        in_flight[k] = in_flight.back();
        in_flight.pop_back();
        progressed = true;
      }

      lock.lock();
      if (!progressed && !in_flight.empty()) {
        // Requests in flight: back off briefly unless new work arrives
        cv.wait_for(lock, std::chrono::microseconds(20),
          [this] { return done || !network_ready.empty(); });
      }
    }
  }
};

} // namespace

namespace ttnte::task {

TaskScheduler::TaskScheduler(
  size_t num_threads, std::optional<std::string> label)
  : label_(label.has_value() ? Label::from_string(label.value())
                             : Label::create_internal()),
    stream_pool_(parallel::StreamPool::create(num_threads)),
    thread_pool_(num_threads, build_worker_init(stream_pool_))
{}

void TaskScheduler::execute(TaskGraph& graph, ExecutionMode mode)
{
  // Lock class from multiple threads trying to execute the same graph
  std::lock_guard<std::mutex> lock(mutex);
  if (mode == ExecutionMode::PUSH)
    execute_push(graph);
  else
    execute_poll(graph);
}

void TaskScheduler::execute_push(TaskGraph& graph)
{
  const auto& plan = graph.push_plan();
  const size_t n = plan.num_tasks;
  if (n == 0)
    return;
  if (!plan.supported) {
    throw utils::runtime_error("ttnte::task::TaskScheduler::execute",
      "ExecutionMode::PUSH supports CPU_ASYNC and NETWORK_* tasks only");
  }
  if (plan.roots.empty()) {
    throw utils::runtime_error(
      "ttnte::task::TaskScheduler::execute", "The task graph has no roots");
  }

  auto run = std::make_shared<PushRun>();
  run->pool = &thread_pool_;
  run->tasks = &graph.get_tasks();
  run->plan = &plan;
  run->pending = std::make_unique<std::atomic<int64_t>[]>(n);
  for (size_t i = 0; i < n; i++)
    run->pending[i].store(plan.num_dependencies[i], std::memory_order_relaxed);
  run->remaining.store(n, std::memory_order_relaxed);

  // Network roots first: no worker runs yet, so network_ready needs no lock
  for (int64_t r : plan.roots)
    if (plan.is_network[r])
      run->network_ready.push_back(r);
  for (int64_t r : plan.roots)
    if (!plan.is_network[r])
      run->enqueue_cpu(r);
  run->drive();

  if (run->incomplete.load()) {
    throw utils::runtime_error("ttnte::task::TaskScheduler::execute",
      "ExecutionMode::PUSH requires CPU payloads that return COMPLETED");
  }
}

void TaskScheduler::execute_poll(TaskGraph& graph)
{

  int tasks_completed = 0;
  const int total_tasks = graph.size();

  while (tasks_completed < total_tasks) {
    bool made_progress = false;

    for (auto& task : graph.get_tasks()) {
      TaskStatus current_status = task.get_status();

      // Check dependencies
      if (current_status == TaskStatus::WAITING && task.check_dependencies()) {
        current_status = TaskStatus::READY;
        task.update_status(current_status);
        made_progress = true;
      }

      // Dispatch READY tasks
      if (current_status == TaskStatus::READY) {

        // Transition out of READY on the main thread before dispatching
        current_status = TaskStatus::RUNNING;
        task.update_status(current_status);
        made_progress = true;

        DeviceTarget target = task.get_target();

        if (target == DeviceTarget::CPU_SYNC ||
            target == DeviceTarget::GPU_SYNC ||
            target == DeviceTarget::NETWORK_SYNC) {

          // Execute blocking payload directly on the main scheduler thread
          task.execute();
          assert(task.get_status() == TaskStatus::COMPLETED);

        } else if (target == DeviceTarget::NETWORK_ASYNC) {

          // Execute MPI tasks inline on the main scheduler thread.
          // MPI calls (MPI_Improbe, MPI_Imrecv, MPI_Test) are not thread-safe
          // unless MPI_THREAD_MULTIPLE is available; running them here keeps
          // all MPI on one thread and avoids data races in OpenMPI internals.
          // The operations themselves are non-blocking, so the scheduler
          // continues dispatching and polling other tasks between retries.
          task.execute();

        } else { // CPU_ASYNC, GPU_ASYNC

          // Hand off the initial dispatch to a background worker
          thread_pool_.push_task([&task]() {
            // The payload performs the work or hardware dispatch, and
            // inherently updates its own atomic status to POLLING or COMPLETED
            task.execute();
          });
        }
      }

      // Fetch status again in case an async thread just finished its dispatch
      current_status = task.get_status();

      // Poll hardware for asynchronous tasks
      if (current_status == TaskStatus::POLLING) {
        // The payload now acts as a non-blocking check
        task.execute();

        // Fetch status one last time for the final count check
        current_status = task.get_status();
      }

      // Mark completed tasks
      if (current_status == TaskStatus::COMPLETED && !task.is_counted()) {
        tasks_completed++;
        task.count();
        made_progress = true;
      }
    }

    // If a full pass of the DAG resulted in no dependencies met, no dispatches,
    // and no completed polls, yield the CPU time slice to the OS.
    if (!made_progress) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
}

} // namespace ttnte::task
