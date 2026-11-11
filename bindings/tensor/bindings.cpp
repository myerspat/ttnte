#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void register_Index(py::module_& m);
void register_Layout(py::module_& m);

// Initialize the tensor module
void init_tensor(py::module_& m)
{
  m.doc() = "Tensor indices and tensor-train layouts";

  // Register classes
  register_Index(m);
  register_Layout(m);
}
