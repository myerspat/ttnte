#include "ttnte/solvers/memory_policy.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_MemoryPolicy(py::module_& m)
{
  using namespace ttnte::solvers;

  // Mirroring the ttnte::solvers namespace structure
  py::enum_<MemoryPolicy>(m, "MemoryPolicy", py::arithmetic())
    .value("RESIDENT", MemoryPolicy::RESIDENT,
      "Operators and state vectors remain on the GPU")
    .value("STATE_RESIDENT", MemoryPolicy::STATE_RESIDENT,
      "Only the state vectors remain on the GPU")
    .value("OPERATOR_RESIDENT", MemoryPolicy::OPERATOR_RESIDENT,
      "Only the operators persist on the GPU")
    .value("OUT_OF_CORE", MemoryPolicy::OUT_OF_CORE,
      "Neither operators nor state vectors persist in GPU memory")
    .export_values();
}
