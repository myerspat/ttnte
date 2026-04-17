#include "ttnte/parallel/thread_pool.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_ThreadPool(py::module_& m)
{
  using namespace ttnte::parallel;

  py::class_<ThreadPool>(m, "ThreadPool")
    // =================================================================
    // Public constructors
    .def(py::init<size_t>(), py::arg("num_threads") = 4)

    // =================================================================
    // Public methods
    // We bind push_task to accept a std::function.
    // pybind11/functional.h handles the conversion from a Python
    // lambda/callable.
    .def(
      "push_python_task",
      [](ThreadPool& self, std::function<void()> f) {
        // We wrap the task to handle the Python GIL if the callable
        // happens to be a Python function.
        self.push_task([f = std::move(f)]() {
          // If this is a Python function, it needs the GIL to run.
          // If it's a pure C++ function, this is a slight overhead
          // but ensures safety.
          py::gil_scoped_acquire acquire;
          f();
        });
      },
      py::arg("func"), "Pushes a task (Python callable) into the thread pool.")

    // Overload for cases where we know we don't need the GIL (higher
    // performance)
    .def(
      "push_task",
      [](ThreadPool& self, std::function<void()> f) {
        self.push_task(std::move(f));
      },
      py::arg("func"),
      "Pushes a C++ task (requires no GIL) into the thread pool.");
}
