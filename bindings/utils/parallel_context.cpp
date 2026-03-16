#include "ttnte/utils/parallel_context.hpp"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_ParallelContext(py::module_& m)
{
  using ParallelContext = ttnte::utils::ParallelContext;

  py::class_<ParallelContext>(m, "ParallelContext")
    .def_static("instance", &ParallelContext::instance,
      py::return_value_policy::reference)
    .def("init", &ParallelContext::init)
    .def("finalize", &ParallelContext::finalize)
    .def_property_readonly("rank", &ParallelContext::rank)
    .def_property_readonly("world_size", &ParallelContext::world_size)
    .def_property_readonly("local_rank", &ParallelContext::local_rank)
    .def_property_readonly("device", [](const ParallelContext& context) {
      return py::cast(context.instance().device());
    });
}
