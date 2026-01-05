#include "ttnte/linalg/contraction_step.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_ContractionStep(py::module_& m)
{
  py::class_<ttnte::linalg::ContractionStep>(m, "ContractionStep")
    .def(py::init<const int64_t&, const int64_t&, const std::vector<int64_t>&,
           const std::vector<int64_t>&,
           const std::optional<std::vector<int64_t>>&>(),
      py::arg("lndim"), py::arg("rndim"), py::arg("ldims"), py::arg("rdims"),
      py::arg("permute") = py::none())
    .def("contract", &ttnte::linalg::ContractionStep::contract,
      py::arg("ltensor"), py::arg("rtensor"));
}
