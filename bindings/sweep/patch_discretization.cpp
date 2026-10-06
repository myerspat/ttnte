#include "ttnte/sweep/patch_discretization.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_PatchDiscretization(py::module_& m)
{
  using namespace ttnte::sweep;

  py::class_<FaceGeometry>(m, "FaceGeometry")
    .def_readonly("dim", &FaceGeometry::dim)
    .def_readonly("is_upper", &FaceGeometry::is_upper)
    .def_readonly("shape", &FaceGeometry::shape)
    .def_readonly("dofs", &FaceGeometry::dofs)
    .def_readonly("normal", &FaceGeometry::normal)
    .def_readonly("weights", &FaceGeometry::weights)
    .def_readonly("area", &FaceGeometry::area)
    .def("upwind_weights", &FaceGeometry::upwind_weights, py::arg("omega"),
      py::arg("outflow"),
      "(D, Qf) clipped, weighted normal components (Omega . n)_+/- w.");

  py::class_<PatchDiscretization, PatchDiscretization::Ptr>(
    m, "PatchDiscretization")
    .def(py::init([](const ttnte::cad::Patch::Ptr& patch) {
      return PatchDiscretization::create(*patch);
    }),
      py::arg("patch"))
    .def("streaming_values", &PatchDiscretization::streaming_values,
      py::arg("omega"))
    .def("to_band", &PatchDiscretization::to_band, py::arg("values"))
    .def("to_dense", &PatchDiscretization::to_dense, py::arg("values"))
    .def("apply_mass", &PatchDiscretization::apply_mass, py::arg("x"))
    .def("local_operator", &PatchDiscretization::local_operator,
      py::arg("omega"), py::arg("sigma_t"),
      "Dense (N, N) local operator sigma_t M - Omega . G + sum_f B_f^+.")
    .def("face", &PatchDiscretization::face, py::arg("dim"),
      py::arg("is_upper"), py::return_value_policy::reference_internal)
    .def_property_readonly("ndim", &PatchDiscretization::ndim)
    .def_property_readonly("shape", &PatchDiscretization::shape)
    .def_property_readonly("num_dofs", &PatchDiscretization::num_dofs)
    .def_property_readonly("num_pairs", &PatchDiscretization::num_pairs)
    .def_property_readonly("kl", &PatchDiscretization::kl)
    .def_property_readonly("ku", &PatchDiscretization::ku)
    .def_property_readonly(
      "mass",
      [](const PatchDiscretization& d) {
        return d.to_dense(d.mass_values().unsqueeze(0))[0];
      },
      "Dense mass matrix.")
    .def_property_readonly(
      "streaming",
      [](const PatchDiscretization& d) { return d.to_dense(d.grad_values()); },
      "Dense (ndim, N, N) streaming matrices (d_i R_v, R_u).")
    .def_property_readonly(
      "basis_integral", &PatchDiscretization::basis_integral);
}
