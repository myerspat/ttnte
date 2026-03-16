#include <torch/extension.h>

void register_ParallelContext(py::module_& m);

PYBIND11_MODULE(utils, m)
{
  m.doc() = "ttnte.utils module";

  register_ParallelContext(m);
}
