#include "ttnte/linalg/gram_schmidt.hpp"
#include "../utils/kwargs_init.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_gram_schmidt(py::module_& m)
{
  using namespace ttnte::linalg;

  py::class_<MGS2Options>(m, "MGS2Options")
    // Keyword arguments only, each an attribute below; defaults from C++
    .def(ttnte::bindings::kwargs_init<MGS2Options>())
    .def_readwrite(
      "eps", &MGS2Options::eps, "Cleanups continue while |q_1^T u| > eps.")
    .def_readwrite("max_projections", &MGS2Options::max_projections,
      "Projections in all (at least 2).")
    .def_readwrite("rtol", &MGS2Options::rtol,
      "Not appended when the remainder is at most rtol ||x||.");

  py::class_<MGS2Result>(m, "MGS2Result")
    .def_readonly("coefficients", &MGS2Result::coefficients)
    .def_readonly("alpha", &MGS2Result::alpha)
    .def_readonly("norm", &MGS2Result::norm)
    .def_readonly("appended", &MGS2Result::appended)
    .def_readonly("orthogonal", &MGS2Result::orthogonal);

  m.def(
    "mgs2a",
    [](const torch::Tensor& Q, torch::Tensor x, const MGS2Options& options) {
      return mgs2a(Q, x, options);
    },
    "Orthogonalize x (n) against the orthonormal rows of Q (r, n) by "
    "Gram-Schmidt with reorthogonalization and adaptive cleanups (MGS2-A): "
    "x = Q^T coefficients + alpha u. x is overwritten in place with the unit "
    "remainder u when appended.",
    py::arg("Q"), py::arg("x"), py::arg("options") = MGS2Options {});
}
