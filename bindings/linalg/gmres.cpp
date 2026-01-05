#include "ttnte/linalg/gmres.hpp"
#include "ttnte/linalg/operator.hpp"
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void register_gmres(py::module_& m)
{
  using Operator = ttnte::linalg::Operator;

  m.def(
    "gmres",
    [](std::shared_ptr<Operator> A, torch::Tensor b,
      const std::optional<torch::Tensor>& x0 = std::nullopt,
      const std::optional<int64_t>& gpu_idx = std::nullopt, double tol = 1e-5,
      double atol = 0.0, int64_t restart = 20,
      std::optional<int64_t> maxiter = std::nullopt,
      const std::string& solve_method = "batched",
      std::optional<py::function> callback = std::nullopt,
      const int64_t& callback_frequency = 1, const bool& verbose = true) {
      // Redirect to Python output
      py::scoped_ostream_redirect stream(
        std::cout, py::module_::import("sys").attr("stdout"));
      return ttnte::linalg::gmres(A, b, x0, gpu_idx, tol, atol, restart,
        maxiter, solve_method, callback, callback_frequency, verbose);
    },
    py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
    py::arg("gpu_idx") = std::nullopt, py::arg("tol") = 1e-5,
    py::arg("atol") = 0.0, py::arg("restart") = 20,
    py::arg("maxiter") = std::nullopt, py::arg("solve_method") = "batched",
    py::arg("callback") = std::nullopt, py::arg("callback_frequency") = 1,
    py::arg("verbose") = true);
}
