#include "ttnte/cad/patch.hpp"
#include "../mesh/mesh_block.hpp"
#include "../utils/label.hpp"
#include <c10/util/SmallVector.h>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;

void register_Patch(py::module_& m)
{
  using Basis = std::vector<ttnte::cad::BSplineBasis>;
  using Patch = ttnte::cad::Patch;
  register_Label<Patch>(m, "Patch");

  auto py_class =
    py::class_<Patch, std::shared_ptr<Patch>>(m, "Patch")
      // =================================================================
      // Public constructors
      .def(py::init([](std::optional<std::string> label) {
        return Patch::create(label);
      }),
        py::arg("label") = py::none())
      .def(py::init([](const torch::Tensor& ctrlpts, const Basis& basis,
                      bool is_rational, std::optional<std::string> label) {
        return Patch::create(ctrlpts, Patch::Basis(basis.begin(), basis.end()),
          is_rational, label);
      }),
        py::arg("ctrlpts"), py::arg("basis"), py::arg("is_rational") = false,
        py::arg("label") = py::none())
      .def(
        py::init([](const torch::Tensor& ctrlpts, const torch::Tensor& weights,
                   const Basis& basis, std::optional<std::string> label) {
          return Patch::create(
            ctrlpts, weights, Patch::Basis(basis.begin(), basis.end()), label);
        }),
        py::arg("ctrlpts"), py::arg("weights"), py::arg("basis"),
        py::arg("label") = py::none())

      // =================================================================
      // Public methods
      .def("is_rational", &Patch::is_rational)
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
      .def(
        "evaluate_basis",
        [](Patch& self, const std::vector<torch::Tensor>& local_coords) {
          return self.evaluate_basis(c10::SmallVector<torch::Tensor, 3>(
            local_coords.begin(), local_coords.end()));
        },
        py::arg("local_coords"))

      // Knot Refinement Methods
      // Knot Insertion
      .def(
        "knot_insert",
        [](Patch& self, const std::vector<torch::Tensor>& new_knots,
          const int64_t reps) {
          return self.knot_insert(c10::SmallVector<torch::Tensor, 3>(
                                    new_knots.begin(), new_knots.end()),
            reps);
        },
        py::arg("new_knots"), py::arg("reps") = 1)
      .def(
        "knot_insert_",
        [](Patch& self, const std::vector<torch::Tensor>& new_knots,
          const int64_t reps) {
          self.knot_insert_(c10::SmallVector<torch::Tensor, 3>(
                              new_knots.begin(), new_knots.end()),
            reps);
        },
        py::arg("new_knots"), py::arg("reps") = 1)

      .def("get_order", &Patch::get_order, py::arg("dim"))
      .def("get_degree", &Patch::get_degree, py::arg("dim"))
      .def("get_ctrlpts_size", &Patch::get_ctrlpts_size, py::arg("dim"))

      .def("set_ctrlptsw", &Patch::set_ctrlptsw, py::arg("ctrlptsw"))
      .def(
        "set_basis",
        [](Patch& self, const Basis& basis) {
          self.set_basis(Patch::Basis(basis.begin(), basis.end()));
        },
        py::arg("basis"))
      .def("set_ctrlpts", &Patch::set_ctrlpts, py::arg("ctrlpts"))
      .def("set_weights", &Patch::set_weights, py::arg("weights"))

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
      .def_property("ctrlptsw", &Patch::get_ctrlptsw, &Patch::set_ctrlptsw)
      .def_property(
        "basis",
        [](const Patch& self) {
          const auto& basis = self.get_basis();
          return Basis(basis.begin(), basis.end());
        },
        [](Patch& self, const Basis& basis) {
          self.set_basis(Patch::Basis(basis.begin(), basis.end()));
        })

      // Tensor and Geometry helpers
      .def_property("ctrlpts", &Patch::get_ctrlpts, &Patch::set_ctrlpts)
      .def_property("weights", &Patch::get_weights, &Patch::set_weights)
      .def_property_readonly("orders", &Patch::get_orders)
      .def_property_readonly("degrees", &Patch::get_degrees)
      .def_property_readonly("ctrlpts_sizes", &Patch::get_ctrlpts_sizes);

  // Register MeshBlock specific methods
  register_MeshBlock(py_class);
}
