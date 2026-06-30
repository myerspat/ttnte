#include "ttnte/parallel/stream_guard.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_StreamGuard(py::module_& m)
{
  using namespace ttnte::parallel;

  // Bind the StreamGuard
  // Note: Since it's non-copyable, pybind11 will handle it via
  // pointer/unique_ptr internally
  py::class_<StreamGuard>(m, "StreamGuard")
    // =================================================================
    // Public constructors
    .def(py::init<c10::Stream>(), py::arg("stream"))

    // =================================================================
    // Operator overloads
    .def("__enter__", [](py::object self) { return self; })
    .def("__exit__", [](StreamGuard& self, py::handle type, py::handle value,
                       py::handle traceback) {
      // The C++ destructor handles the heavy lifting when this object
      // is garbage collected or goes out of scope at the end of the 'with'
      // block
    });
}
