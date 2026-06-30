#include "ttnte/task/task_scheduler.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_TaskScheduler(py::module_& m)
{
  using namespace ttnte::task;

  py::class_<TaskScheduler>(m, "TaskScheduler")
    // =================================================================
    // Public constructor
    .def(py::init<size_t, std::optional<std::string>>(),
      py::arg("num_threads") = 4, py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def("execute", &TaskScheduler::execute, py::arg("graph"),
      py::call_guard<py::gil_scoped_release>())

    .def("get_label", &TaskScheduler::get_label,
      py::return_value_policy::reference_internal)

    // =================================================================
    // Public getters / setters
    .def_property_readonly("label", &TaskScheduler::get_label);
}
