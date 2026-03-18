#include "ttnte/cad/patch.hpp"
#include "../utils/label.hpp"
#include "ttnte/xs/material.hpp"
#include <c10/util/SmallVector.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void register_Patch(py::module_& m)
{
  using Basis = std::vector<ttnte::cad::BSplineBasis>;
  using Patch = ttnte::cad::Patch;
  register_Label<Patch>(m, "Patch");

  py::class_<Patch>(m, "Patch")
    // =================================================================
    // Public constructors
    .def(py::init<std::optional<std::string>>(), py::arg("label") = py::none())
    .def(py::init([](const torch::Tensor& ctrlpts, const Basis& basis,
                    bool is_rational, std::optional<std::string> label) {
      return Patch(
        ctrlpts, Patch::Basis(basis.begin(), basis.end()), is_rational, label);
    }),
      py::arg("ctrlpts"), py::arg("basis"), py::arg("is_rational") = false,
      py::arg("label") = py::none())
    .def(py::init([](const torch::Tensor& ctrlpts, const torch::Tensor& weights,
                    const Basis& basis, std::optional<std::string> label) {
      return Patch(
        ctrlpts, weights, Patch::Basis(basis.begin(), basis.end()), label);
    }),
      py::arg("ctrlpts"), py::arg("weights"), py::arg("basis"),
      py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def("validate", &Patch::validate)
    .def("invalidate", &Patch::invalidate)
    .def("is_valid", &Patch::is_valid)
    .def("is_rational", &Patch::is_rational)
    .def("clone", &Patch::clone)
    .def("evaluate",
      static_cast<torch::Tensor (Patch::*)(const torch::Tensor&)>(
        &Patch::evaluate),
      py::arg("local_coords"))
    .def(
      "evaluate",
      [](Patch& self, const std::vector<torch::Tensor>& local_coords) {
        return self.evaluate(c10::SmallVector<torch::Tensor, 3>(
          local_coords.begin(), local_coords.end()));
      },
      py::arg("local_coords"))

    .def("get_order", &Patch::get_order, py::arg("dim"))
    .def("get_degree", &Patch::get_degree, py::arg("dim"))
    .def("get_ctrlpts_size", &Patch::get_ctrlpts_size, py::arg("dim"))
    .def("get_bbox", &Patch::get_bbox, py::arg("epsilon") = 0.0)
    .def("get_boundary", &Patch::get_boundary, py::arg("dim"),
      py::arg("is_upper"), py::arg("clone") = true)

    .def("set_ctrlptsw", &Patch::set_ctrlptsw, py::arg("ctrlptsw"),
      py::arg("clone") = true)
    .def(
      "set_basis",
      [](Patch& self, const Basis& basis, bool clone = true) {
        self.set_basis(Patch::Basis(basis.begin(), basis.end()), clone);
      },
      py::arg("basis"), py::arg("clone") = true)
    .def("set_ctrlpts", &Patch::set_ctrlpts, py::arg("ctrlpts"),
      py::arg("clone") = true)
    .def("set_weights", &Patch::set_weights, py::arg("weights"),
      py::arg("clone") = true)

    // =================================================================
    // Public overloads
    .def("__call__",
      static_cast<torch::Tensor (Patch::*)(const torch::Tensor&)>(
        &Patch::evaluate),
      py::arg("local_coords"))
    .def(
      "__call__",
      [](Patch& self, const std::vector<torch::Tensor>& local_coords) {
        return self.evaluate(c10::SmallVector<torch::Tensor, 3>(
          local_coords.begin(), local_coords.end()));
      },
      py::arg("local_coords"))
    .def("__repr__",
      [](const Patch& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      })

    // Properties (Getters/Setters)
    .def_property("label", &Patch::get_label,
      py::overload_cast<const std::string&>(&Patch::set_label))
    .def_property("ctrlptsw", &Patch::get_ctrlptsw,
      [](Patch& self, const torch::Tensor& ctrlptsw) {
        self.set_ctrlptsw(ctrlptsw, true);
      })
    .def_property(
      "basis",
      [](const Patch& self) {
        const auto& basis = self.get_basis();
        return Basis(basis.begin(), basis.end());
      },
      [](Patch& self, const Basis& basis) {
        self.set_basis(Patch::Basis(basis.begin(), basis.end()), true);
      })

    // Tensor and Geometry helpers
    .def_property("ctrlpts", &Patch::get_ctrlpts,
      [](Patch& self, const torch::Tensor& ctrlpts) {
        self.set_ctrlpts(ctrlpts, true);
      })
    .def_property("weights", &Patch::get_weights,
      [](Patch& self, const torch::Tensor& weights) {
        self.set_weights(weights, true);
      })
    .def_property("fill_id", &Patch::get_fill_id, &Patch::set_fill_id)
    .def_property("fill", &Patch::get_fill<ttnte::xs::Material>,
      &Patch::set_fill<ttnte::xs::Material>)
    .def_property_readonly("orders", &Patch::get_orders)
    .def_property_readonly("degrees", &Patch::get_degrees)
    .def_property_readonly("ctrlpts_sizes", &Patch::get_ctrlpts_sizes)
    .def_property_readonly("device", &Patch::get_device)
    .def_property_readonly("dtype", &Patch::get_dtype);
}
