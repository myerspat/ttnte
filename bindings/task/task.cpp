#include "ttnte/task/task.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_Task(py::module_& m)
{
  using namespace ttnte::task;

  // Bind TaskStatus
  py::enum_<TaskStatus>(m, "TaskStatus")
    .value("WAITING", TaskStatus::WAITING)
    .value("READY", TaskStatus::READY)
    .value("RUNNING", TaskStatus::RUNNING)
    .value("POLLING", TaskStatus::POLLING)
    .value("COMPLETED", TaskStatus::COMPLETED)
    .export_values();

  // Bind DeviceTarget
  py::enum_<DeviceTarget>(m, "DeviceTarget")
    .value("CPU_SYNC", DeviceTarget::CPU_SYNC)
    .value("CPU_ASYNC", DeviceTarget::CPU_ASYNC)
    .value("GPU_SYNC", DeviceTarget::GPU_SYNC)
    .value("GPU_ASYNC", DeviceTarget::GPU_ASYNC)
    .value("NETWORK_SYNC", DeviceTarget::NETWORK_SYNC)
    .value("NETWORK_ASYNC", DeviceTarget::NETWORK_ASYNC)
    .export_values();

  // Bind Task
  py::class_<Task>(m, "Task")
    .def(py::init<const DeviceTarget&, std::optional<std::string>>(),
      py::arg("target"), py::arg("label") = py::none())

    .def_property_readonly("target", &Task::get_target)
    .def_property_readonly("status", &Task::get_status)
    .def_property_readonly("label", &Task::get_label);
}
