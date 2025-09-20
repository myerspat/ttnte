#include "ttnte/linalg/eig.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void register_eig(py::module_& m)
{
  using LinearSolverOptions = ttnte::linalg::LinearSolverOptions;

  py::class_<LinearSolverOptions>(m, "LinearSolverOptions")
    .def(py::init<>())
    .def_readwrite("tol", &LinearSolverOptions::tol)
    .def_readwrite("atol", &LinearSolverOptions::atol)
    .def_readwrite("restart", &LinearSolverOptions::restart)
    .def_readwrite("maxiter", &LinearSolverOptions::maxiter)
    .def_readwrite("solve_method", &LinearSolverOptions::solve_method)
    .def_readwrite("gpu_idx", &LinearSolverOptions::gpu_idx)
    .def_readwrite("callback", &LinearSolverOptions::callback)
    .def_readwrite(
      "callback_frequency", &LinearSolverOptions::callback_frequency)
    .def_readwrite("verbose", &LinearSolverOptions::verbose);

  m.def("power", &ttnte::linalg::power, py::arg("T"), py::arg("F"),
    py::arg("psi0") = std::nullopt, py::arg("tol") = 1e-8,
    py::arg("maxiter") = 100, py::arg("gpu_idx") = std::nullopt,
    py::arg("callback") = std::nullopt, py::arg("callback_frequency") = 1,
    py::arg("verbose") = true, py::arg("lsoptions") = LinearSolverOptions());
}
