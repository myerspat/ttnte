#include "ttnte/math/quadrature_set.hpp"
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

void register_quadrature_set(py::module_& m)
{
  using namespace ttnte::math;

  // 1. Register the Symmetry Enum
  py::enum_<Symmetry>(m, "Symmetry")
    .value("NONE", Symmetry::NONE)
    .value("XY_PLANE", Symmetry::XY_PLANE)
    .export_values();

  // Base Class Binding
  py::class_<QuadratureSet, QuadratureSet::Ptr>(m, "QuadratureSet")
    // =================================================================
    // Public methods
    .def("is_tensor_product", &QuadratureSet::is_tensor_product)

    // Custom robust lambda for the TensorOptions overload
    .def("to_",
      py::overload_cast<const torch::Device&, const torch::ScalarType&>(
        &QuadratureSet::to_),
      py::arg("device"), py::arg("dtype"))
    .def(
      "to_", py::overload_cast<const torch::ScalarType&>(&QuadratureSet::to_))
    .def("to_", py::overload_cast<const torch::Device&>(&QuadratureSet::to_))

    // =================================================================
    // Public Getters / Setters
    .def("get_points", &QuadratureSet::get_points,
      py::return_value_policy::reference_internal)
    .def("get_weights", &QuadratureSet::get_weights,
      py::return_value_policy::reference_internal)
    .def("get_num_dofs", &QuadratureSet::get_num_dofs)
    .def("get_weighting_factor", &QuadratureSet::get_weighting_factor)
    .def("get_symmetry", &QuadratureSet::get_symmetry) // New C++ binding

    .def_property_readonly("points", &QuadratureSet::get_points,
      py::return_value_policy::reference_internal)
    .def_property_readonly("weights", &QuadratureSet::get_weights,
      py::return_value_policy::reference_internal)
    .def_property_readonly("num_dofs", &QuadratureSet::get_num_dofs)
    .def_property_readonly(
      "weighting_factor", &QuadratureSet::get_weighting_factor)
    .def_property_readonly(
      "symmetry", &QuadratureSet::get_symmetry); // New Python property

  // 1-D Derived Class Binding
  py::class_<QuadratureSet1D, QuadratureSet, QuadratureSet1D::Ptr>(
    m, "QuadratureSet1D")
    // =================================================================
    // Public constructors
    .def(py::init([](const torch::Tensor& points, const torch::Tensor& weights,
                    double weighting_factor, Symmetry symmetry) {
      return QuadratureSet1D::create(
        points, weights, weighting_factor, symmetry);
    }),
      py::arg("points"), py::arg("weights"), py::arg("weighting_factor"),
      py::arg("symmetry") = Symmetry::NONE)
    // =================================================================
    // Public methods
    .def_static("gauss_legendre", &QuadratureSet1D::gauss_legendre,
      py::arg("n"), py::arg("weighting_factor") = 2.0)
    .def_static("gauss_chebyshev", &QuadratureSet1D::gauss_chebyshev,
      py::arg("n"), py::arg("weighting_factor") = 2.0 * std::numbers::pi);

  // Product Quadrature Derived Class Binding
  py::class_<ProductQuadrature, QuadratureSet, ProductQuadrature::Ptr>(
    m, "ProductQuadrature")
    // =================================================================
    // Public constructors
    .def(py::init([](const std::vector<QuadratureSet1D::Ptr>& quads,
                    double weighting_factor, Symmetry symmetry) {
      return ProductQuadrature::create(
        ProductQuadrature::Quads(quads.cbegin(), quads.cend()),
        weighting_factor, symmetry);
    }),
      py::arg("quads"), py::arg("weighting_factor") = 1.0,
      py::arg("symmetry") = Symmetry::NONE)

    // =================================================================
    // Public methods (Fixed arguments match your header defaults exactly)
    .def_static("gauss_legendre_chebyshev",
      &ProductQuadrature::gauss_legendre_chebyshev, py::arg("n_polar"),
      py::arg("n_azimuthal"), py::arg("ndim") = 3,
      py::arg("weighting_factor") = 1.0)

    // =================================================================
    // Public Getters / Setters
    .def("get_quads",
      [](const ProductQuadrature& self) {
        const auto& quads = self.get_quads();
        return std::vector<QuadratureSet1D::Ptr>(quads.begin(), quads.end());
      })
    .def("get_factored_points",
      [](const ProductQuadrature& self) {
        const auto& p = self.get_factored_points();
        return std::vector<torch::Tensor>(p.begin(), p.end());
      })
    .def("get_factored_weights",
      [](const ProductQuadrature& self) {
        const auto& w = self.get_factored_weights();
        return std::vector<torch::Tensor>(w.begin(), w.end());
      })
    .def("get_ndim", &ProductQuadrature::get_ndim)

    .def_property_readonly("quads",
      [](const ProductQuadrature& self) {
        const auto& quads = self.get_quads();
        return std::vector<QuadratureSet1D::Ptr>(quads.begin(), quads.end());
      })
    .def_property_readonly("factored_points",
      [](const ProductQuadrature& self) {
        const auto& p = self.get_factored_points();
        return std::vector<torch::Tensor>(p.begin(), p.end());
      })
    .def_property_readonly("factored_weights",
      [](const ProductQuadrature& self) {
        const auto& w = self.get_factored_weights();
        return std::vector<torch::Tensor>(w.begin(), w.end());
      })
    .def_property_readonly("ndim", &ProductQuadrature::get_ndim);
}
