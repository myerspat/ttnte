#include "ttnte/cad/bspline_basis.hpp"
#include "basis.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_BSplineBasis(py::module_& m)
{
  using namespace ttnte::cad;
  using Base = Basis<BSplineBasis>;
  register_Basis<BSplineBasis>(m, "BSplineBasis");

  py::class_<BSplineBasis, Basis<BSplineBasis>>(m, "BSplineBasis")
    // =================================================================
    // Public constructors
    .def(py::init<torch::Tensor, int64_t, std::optional<std::string>>(),
      py::arg("knotvector"), py::arg("degree"), py::arg("label") = py::none())
    .def(py::init<const torch::Tensor&, int64_t, const Base::Label&>(),
      py::arg("knotvector"), py::arg("degree"), py::arg("label"))

    // =================================================================
    // Public methods
    .def("is_valid", &BSplineBasis::is_valid)
    .def("normalize_knotvector", &BSplineBasis::normalize_knotvector)
    .def("find_spans", &BSplineBasis::find_spans, py::arg("u"))
    .def("evaluate",
      py::overload_cast<const torch::Tensor&, const int64_t&>(
        &BSplineBasis::evaluate, py::const_),
      py::arg("u"), py::arg("derivative_order") = 0)
    .def("evaluate",
      py::overload_cast<const torch::Tensor&, const torch::Tensor&,
        const int64_t&>(&BSplineBasis::evaluate, py::const_),
      py::arg("u"), py::arg("spans"), py::arg("derivative_order") = 0)
    .def("evaluate_all",
      py::overload_cast<const torch::Tensor&, const int64_t&>(
        &BSplineBasis::evaluate_all, py::const_),
      py::arg("u"), py::arg("derivative_order") = 0)
    .def("evaluate_all",
      py::overload_cast<const torch::Tensor&, const torch::Tensor&,
        const int64_t&>(&BSplineBasis::evaluate_all, py::const_),
      py::arg("u"), py::arg("spans"), py::arg("derivative_order") = 0)

    // =================================================================
    // Overloads
    .def("__call__",
      py::overload_cast<const torch::Tensor&, const int64_t&>(
        &BSplineBasis::evaluate, py::const_),
      py::arg("u"), py::arg("derivative_order") = 0)
    .def("__call__",
      py::overload_cast<const torch::Tensor&, const torch::Tensor&,
        const int64_t&>(&BSplineBasis::evaluate, py::const_),
      py::arg("u"), py::arg("spans"), py::arg("derivative_order") = 0)
    .def("__repr__",
      [](const BSplineBasis& b) {
        std::stringstream ss;
        ss << b;
        return ss.str();
      })

    // =================================================================
    // Public Getters / Setters
    .def_property_readonly("degree", &BSplineBasis::get_degree)
    .def_property_readonly("knotvector", &BSplineBasis::get_knotvector)
    .def_property_readonly("order", &BSplineBasis::get_order)
    .def_property_readonly("size", &BSplineBasis::get_size)
    .def_property_readonly("unique_knots_and_multiplicity",
      &BSplineBasis::get_unique_knots_and_multiplicity)
    .def_property_readonly("unique_knots", &BSplineBasis::get_unique_knots)
    .def_property_readonly("multiplicity", &BSplineBasis::get_multiplicity);
}
