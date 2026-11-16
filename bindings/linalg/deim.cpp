#include "ttnte/linalg/deim.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_deim(py::module_& m)
{
  using namespace ttnte::linalg;

  m.def("deim_select", &deim_select, py::arg("V"),
    "DEIM row selection of a basis V (n, r) (Han et al. 2026, Alg. B.1): r "
    "rows, in selection order.");
  m.def("qdeim_select", &qdeim_select, py::arg("V"),
    "Q-DEIM row selection: the first r column pivots of a column-pivoted QR "
    "of V^T (LAPACK dgeqp3).");
  m.def("weighted_orthonormal", &weighted_orthonormal, py::arg("L"),
    py::arg("w"),
    "(V, R) with L = V R and V^T diag(w) V = I (Han et al. 2026, eq. 3.5).");
  m.def("lebesgue_constant", &lebesgue_constant, py::arg("V"), py::arg("I"),
    py::arg("w") = py::none(),
    "Norm of the interpolation projector at rows I: ||V_o(I, :)^+||_2, V_o an "
    "orthonormal basis of span(V), or of span(diag(sqrt w) V) with weights w "
    "(the norm ||diag(sqrt w) g||_2).");
}
