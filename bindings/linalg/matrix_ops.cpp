#include "ttnte/linalg/matrix_ops.hpp"
#include <limits>
#include <torch/extension.h>

namespace py = pybind11;

void register_matrix_ops(py::module_& m)
{
  using namespace ttnte::linalg;

  m.def("truncated_svd", &truncated_svd,
    "Compute a truncated singular value decomposition (SVD).", py::arg("A"),
    py::arg("delta") = 0.0,
    py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
    py::arg("with_normalize") = false,
    py::call_guard<py::gil_scoped_release>());
}
