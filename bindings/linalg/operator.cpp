#include "ttnte/linalg/operator.hpp"

#include "ttnte/linalg/fission_operator.hpp"
#include "ttnte/linalg/linear_operator.hpp"
#include "ttnte/linalg/scatter_operator.hpp"
#include "ttnte/linalg/sparse_operator.hpp"
#include "ttnte/linalg/tt_operator.hpp"

namespace py = pybind11;

void register_Operator(py::module_& m)
{
  using Operator = ttnte::linalg::Operator;

  py::class_<Operator, std::shared_ptr<Operator>>(m, "Operator")
    .def("__matmul__", &Operator::apply)
    .def("__add__", &Operator::operator+)
    .def("__sub__",
      [](const std::shared_ptr<ttnte::linalg::Operator> self,
        const std::shared_ptr<ttnte::linalg::LinearOperator> other) {
        return self->operator-(other);
      })
    .def("__sub__",
      [](const std::shared_ptr<ttnte::linalg::Operator> self,
        const std::shared_ptr<ttnte::linalg::TTOperator> other) {
        return self->operator-(other);
      })
    .def("__sub__",
      [](const std::shared_ptr<ttnte::linalg::Operator> self,
        const std::shared_ptr<ttnte::linalg::SparseOperator> other) {
        return self->operator-(other);
      })
    .def("__sub__",
      [](const std::shared_ptr<ttnte::linalg::Operator> self,
        const std::shared_ptr<ttnte::linalg::ScatterOperator> other) {
        return self->operator-(other);
      })
    .def("__sub__",
      [](const std::shared_ptr<ttnte::linalg::Operator> self,
        const std::shared_ptr<ttnte::linalg::FissionOperator> other) {
        return self->operator-(other);
      })
    .def("matvec", &Operator::apply)
    .def("apply", &Operator::apply)
    .def("cuda", &Operator::cuda)
    .def("cpu", &Operator::cpu)
    .def_property_readonly("input_shape", &Operator::input_shape)
    .def_property_readonly("output_shape", &Operator::output_shape)
    .def_property_readonly("nelements", &Operator::nelements)
    .def_property_readonly("compression", &Operator::compression);
}
