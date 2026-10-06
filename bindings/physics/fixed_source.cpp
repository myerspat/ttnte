#include "ttnte/physics/fixed_source.hpp"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_FixedSource(py::module_& m)
{
  using namespace ttnte::physics;

  py::class_<FixedSource>(m, "FixedSource")
    .def(py::init(
           [](std::optional<
                std::function<torch::Tensor(const std::vector<torch::Tensor>&)>>
                function,
             std::optional<torch::Tensor> isotropic_strength,
             std::optional<torch::Tensor> projected_isotropic_source) {
             FixedSource source;
             source.function = std::move(function);
             source.isotropic_strength = std::move(isotropic_strength);
             source.projected_isotropic_source =
               std::move(projected_isotropic_source);
             return source;
           }),
      py::arg("function") = py::none(),
      py::arg("isotropic_strength") = py::none(),
      py::arg("projected_isotropic_source") = py::none(),
      "A fixed (static, non-eigenvalue) source specification: an arbitrary "
      "function of physical-space/angle/energy sample points (e.g. for "
      "MMS), and/or a per-group isotropic strength in physical units "
      "(e.g. neutrons/cm^3/s for a volumetric source, or an incident flux "
      "for a boundary source), and/or an already-projected, "
      "angle-independent load vector b_{i,g} = int q_g(x) R_i(x) dV given as "
      "a dense tensor of shape (control points per parametric dimension..., "
      "energy groups), broadcast uniformly over ordinates.")

    .def_property(
      "function", [](const FixedSource& self) { return self.function; },
      [](FixedSource& self,
        std::optional<
          std::function<torch::Tensor(const std::vector<torch::Tensor>&)>>
          function) { self.function = std::move(function); })
    .def_property(
      "isotropic_strength",
      [](const FixedSource& self) { return self.isotropic_strength; },
      [](FixedSource& self, std::optional<torch::Tensor> strength) {
        self.isotropic_strength = std::move(strength);
      })
    .def_property(
      "projected_isotropic_source",
      [](const FixedSource& self) { return self.projected_isotropic_source; },
      [](FixedSource& self, std::optional<torch::Tensor> source) {
        self.projected_isotropic_source = std::move(source);
      })
    .def_property_readonly(
      "defined", [](const FixedSource& self) { return self.defined(); });
}
