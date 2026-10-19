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
  m.def("gram_truncated_svd", &gram_truncated_svd,
    "Truncated SVD of a tall or wide matrix through its Gram matrix (falls "
    "back to truncated_svd below gram_floor or for near-square matrices).",
    py::arg("A"), py::arg("delta") = 0.0,
    py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
    py::arg("with_normalize") = false, py::arg("gram_floor") = 1e-7,
    py::call_guard<py::gil_scoped_release>());
}
