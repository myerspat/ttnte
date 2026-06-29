#include "ttnte/linalg/source.hpp"
#include <limits>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_Source(py::module_& m)
{
  using namespace ttnte::linalg;

  py::class_<Source, std::shared_ptr<Source>>(m, "Source",
    "Fixed patch-local source term. The source state is static and packed "
    "into the LinearSystem flat buffer.")
    .def(py::init([](State state) { return Source::create(std::move(state)); }),
      py::arg("state") = State(),
      "Construct a fixed Source from a state vector.")
    .def_property_readonly(
      "state", &Source::get_state, "The source state vector.")
    .def_property_readonly("device", &Source::get_device,
      "Device of the static data packed into the flat buffer.")
    .def_property_readonly("dtype", &Source::get_dtype,
      "Scalar type of the static data packed into the flat buffer.")
    .def_property_readonly("is_eigenvalue", &Source::is_eigenvalue,
      "True if this source depends on the current solution iterate.")
    .def(
      "update",
      [](Source& src, const State& eigvec, double eps, int64_t max_rank) {
        src.update(eigvec, eps, max_rank);
      },
      py::arg("eigvec"), py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>(),
      "Recompute state from the current solution iterate (no-op for fixed "
      "sources).");
}

void register_EigenSource(py::module_& m)
{
  using namespace ttnte::linalg;

  py::class_<EigenSource, Source, std::shared_ptr<EigenSource>>(m,
    "EigenSource",
    "Source term for a generalized eigenvalue problem. The operator is static "
    "and packed into the LinearSystem flat buffer; the source state is "
    "recomputed each iteration via update().")
    .def(
      py::init([](Operator op) { return EigenSource::create(std::move(op)); }),
      py::arg("op"), "Construct an EigenSource from the fission operator.")
    .def_property_readonly("op", &EigenSource::get_op, "The fission operator.")
    .def_property_readonly("total_source", &EigenSource::get_total_source,
      "Integral of the unscaled fission source, cached by the most recent "
      "update() call.")
    .def_property("eigval", &EigenSource::get_eigval, &EigenSource::set_eigval,
      "Scale factor applied to the source state. Set by the driver before each "
      "DAG execution.")
    .def(
      "update",
      [](EigenSource& src, const State& eigvec, double eps, int64_t max_rank) {
        src.update(eigvec, eps, max_rank);
      },
      py::arg("eigvec"), py::arg("eps") = 1e-12,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>(),
      "Compute state = mv(op, eigvec), round, cache the element sum as "
      "total_source, then scale state in-place by eigval.");
}
