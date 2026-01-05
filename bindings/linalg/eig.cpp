#include "ttnte/linalg/eig.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void register_eig(py::module_& m)
{
  using LinearSolverOptions = ttnte::linalg::LinearSolverOptions;
  using Operator = ttnte::linalg::Operator;

  py::class_<LinearSolverOptions>(m, "LinearSolverOptions")
    .def(
      py::init<double, double, int64_t, int64_t, std::string,
        std::optional<py::function>, std::optional<int64_t>, int64_t, bool>(),
      py::arg("tol") = 1e-10, py::arg("atol") = 0.0, py::arg("restart") = 100,
      py::arg("maxiter") = 5, py::arg("solve_method") = "batched",
      py::arg("callback") = std::nullopt, py::arg("gpu_idx") = std::nullopt,
      py::arg("callback_frequency") = 1, py::arg("verbose") = false)
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

  m.def(
    "power",
    [](std::shared_ptr<Operator> T, std::shared_ptr<Operator> F,
      const std::optional<torch::Tensor>& psi0 = std::nullopt,
      const double& tol = 1e-8, const int64_t& maxiter = 100,
      const std::optional<int64_t>& gpu_idx = std::nullopt,
      const std::optional<py::function>& callback = std::nullopt,
      const int64_t& callback_frequency = 1, const bool& verbose = true,
      const LinearSolverOptions& lsoptions = LinearSolverOptions()) {
      // Redirect to Python output
      py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));
      return ttnte::linalg::power(T, F, psi0, tol, maxiter, gpu_idx, callback,
        callback_frequency, verbose, lsoptions);
    },
    py::arg("T"), py::arg("F"), py::arg("psi0") = std::nullopt,
    py::arg("tol") = 1e-8, py::arg("maxiter") = 100,
    py::arg("gpu_idx") = std::nullopt, py::arg("callback") = std::nullopt,
    py::arg("callback_frequency") = 1, py::arg("verbose") = true,
    py::arg("lsoptions") = LinearSolverOptions());
}
