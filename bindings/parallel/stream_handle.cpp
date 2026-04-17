#include "ttnte/parallel/stream_handle.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_StreamHandle(py::module_& m)
{
  using namespace ttnte::parallel;

  // Bind StreamHandle struct
  py::class_<StreamHandle>(m, "StreamHandle")
    // =================================================================
    // Public constructors
    .def(py::init<c10::Stream>(), py::arg("stream"))

    // =================================================================
    // Public methods
    .def("guard", [](StreamHandle& self) { return StreamGuard(self.stream); })

    // =================================================================
    // Public getters / setters
    .def_readwrite("stream", &StreamHandle::stream);
}
