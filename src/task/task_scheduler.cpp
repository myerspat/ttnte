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
  if (!plan.all_cpu_async) {
    throw utils::runtime_error("ttnte::task::TaskScheduler::execute",
      "ExecutionMode::PUSH requires every task to be CPU_ASYNC");
  }
  if (plan.roots.empty()) {
    throw utils::runtime_error(
      "ttnte::task::TaskScheduler::execute", "The task graph has no roots");
  }

  // Shared by the workers; kept alive by every queued closure so a worker
  // still unwinding after the last completion never touches freed memory.
  struct State {
    TaskGraph::TaskDeque* tasks;
    const TaskGraph::PushPlan* plan;
    std::unique_ptr<std::atomic<int64_t>[]> pending;
    std::atomic<size_t> remaining;
    std::atomic<bool> incomplete {false};
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
  };
  auto state = std::make_shared<State>();
  state->tasks = &graph.get_tasks();
  state->plan = &plan;
  state->pending = std::make_unique<std::atomic<int64_t>[]>(n);
  for (size_t i = 0; i < n; i++)
    state->pending[i].store(
      plan.num_dependencies[i], std::memory_order_relaxed);
  state->remaining.store(n, std::memory_order_relaxed);

  // Run task i, then keep running one released successor on this thread
  // (data locality: it consumes what i just produced) and queue the others.
  auto run = std::make_shared<std::function<void(int64_t)>>();
  *run = [this, state, weak = std::weak_ptr<std::function<void(int64_t)>>(run)](
           int64_t i) {
    auto self = weak.lock();
    while (i >= 0) {
      Task& task = (*state->tasks)[i];
      task.update_status(TaskStatus::RUNNING);
      task.execute();
      if (task.get_status() == TaskStatus::COMPLETED)
        task.count();
      else
        state->incomplete.store(true, std::memory_order_relaxed);

      int64_t next = -1;
      for (int64_t c : state->plan->successors[i]) {
        if (state->pending[c].fetch_sub(1, std::memory_order_acq_rel) == 1) {
          if (next < 0)
            next = c;
          else
            thread_pool_.push_task([self, c]() { (*self)(c); });
        }
      }
      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(state->done_mutex);
        state->done = true;
        state->done_cv.notify_one();
      }
      i = next;
    }
  };

  for (int64_t r : plan.roots)
    thread_pool_.push_task([run, r]() { (*run)(r); });

  std::unique_lock<std::mutex> lock(state->done_mutex);
  state->done_cv.wait(lock, [&state] { return state->done; });
  lock.unlock();

  if (state->incomplete.load()) {
    throw utils::runtime_error("ttnte::task::TaskScheduler::execute",
      "ExecutionMode::PUSH requires payloads that return COMPLETED");
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
