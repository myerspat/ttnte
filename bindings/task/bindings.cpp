#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_Task(py::module_& m);
void register_TaskGraph(py::module_& m);
void register_TaskScheduler(py::module_& m);

// Initialize the task module
void init_task(py::module_& m)
{
  m.doc() = "Task module for the parallel solver";

  // Register classes
  register_Task(m);
  register_TaskGraph(m);
  register_TaskScheduler(m);
}
