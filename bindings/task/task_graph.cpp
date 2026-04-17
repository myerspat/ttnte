#include "ttnte/task/task_graph.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_TaskGraph(py::module_& m)
{
  using namespace ttnte::task;

  py::class_<TaskGraph>(m, "TaskGraph")
    // =================================================================
    // Public constructor
    .def(py::init<std::optional<std::string>>(), py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def("create_task", &TaskGraph::create_task, py::arg("target") = py::none(),
      py::arg("label") = py::none(),
      py::return_value_policy::reference_internal)
    .def("add_dependency", &TaskGraph::add_dependency, py::arg("dependent"),
      py::arg("dependency"))
    .def("clear", &TaskGraph::clear)

    .def("get_tasks", &TaskGraph::get_tasks,
      py::return_value_policy::reference_internal)

    // =================================================================
    // Public overloads
    .def("__len__", &TaskGraph::size)

    // =================================================================
    // Public getters / setters
    .def_property_readonly("size", &TaskGraph::size);
}
