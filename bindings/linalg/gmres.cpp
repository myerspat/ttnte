#include "ttnte/linalg/gmres.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void register_gmres(py::module_& m)
{
  m.def("gmres", &ttnte::linalg::gmres, py::arg("A"), py::arg("b"),
    py::arg("x0") = std::nullopt, py::arg("gpu_idx") = std::nullopt,
    py::arg("tol") = 1e-5, py::arg("atol") = 0.0, py::arg("restart") = 20,
    py::arg("maxiter") = std::nullopt, py::arg("solve_method") = "batched",
    py::arg("callback") = std::nullopt, py::arg("callback_frequency") = 1,
    py::arg("verbose") = true);
}
