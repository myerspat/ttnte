#include "ttnte/math/special.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_special(py::module_& m)
{
  using namespace ttnte::math::special;

  // Bind the HarmonicComponent Enum Class
  py::enum_<HarmonicComponent>(m, "HarmonicComponent")
    .value("EVEN", HarmonicComponent::EVEN)
    .value("ODD", HarmonicComponent::ODD)
    .value("BOTH", HarmonicComponent::BOTH)
    .export_values();

  // Bind legendre overloads
  m.def("legendre",
    py::overload_cast<uint32_t, const torch::Tensor&, bool>(&legendre),
    py::arg("l"), py::arg("x"), py::arg("skip_checks"));

  // Bind assoc_legendre Overloads
  m.def("assoc_legendre",
    py::overload_cast<uint32_t, uint32_t, const torch::Tensor&, bool>(
      &assoc_legendre),
    py::arg("l"), py::arg("m"), py::arg("x"), py::arg("skip_checks") = false,
    "Computes associated Legendre polynomials for a specific l and m degree "
    "across a tensor grid x.");

  m.def("assoc_legendre",
    py::overload_cast<const torch::Tensor&, const torch::Tensor&,
      const torch::Tensor&, bool>(&assoc_legendre),
    py::arg("l"), py::arg("m"), py::arg("x"), py::arg("skip_checks") = false,
    "Computes associated Legendre polynomials where degrees l and m are "
    "varying tensor inputs matching grid x.");

  // Bind sph_harm Overloads
  m.def("sph_harm",
    py::overload_cast<uint32_t, uint32_t, const torch::Tensor&,
      const torch::Tensor&, HarmonicComponent, bool>(&sph_harm),
    py::arg("l"), py::arg("m"), py::arg("mu"), py::arg("gamma"),
    py::arg("component") = HarmonicComponent::BOTH,
    py::arg("skip_checks") = false,
    "Evaluates real spherical harmonics for a specific l and m order across mu "
    "and gamma coordinate tensors.");

  m.def("sph_harm",
    py::overload_cast<const torch::Tensor&, const torch::Tensor&,
      const torch::Tensor&, const torch::Tensor&, HarmonicComponent, bool>(
      &sph_harm),
    py::arg("l"), py::arg("m"), py::arg("mu"), py::arg("gamma"),
    py::arg("component") = HarmonicComponent::BOTH,
    py::arg("skip_checks") = false,
    "Evaluates real spherical harmonics where degrees l, m, and the component "
    "layout match coordinate inputs.");

  m.def("sph_harm",
    py::overload_cast<const torch::Tensor&, const torch::Tensor&,
      const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, bool>(
      &sph_harm),
    py::arg("l"), py::arg("m"), py::arg("mu"), py::arg("gamma"),
    py::arg("component"), py::arg("skip_checks") = false,
    "Evaluates real spherical harmonics where 'component' is passed as a 1-D "
    "boolean mask tensor.");
}
