#include "ttnte/linalg/tt_ops.hpp"
#include <limits>
#include <optional>
#include <string>
#include <torch/extension.h>

namespace py = pybind11;

void register_tt_ops(py::module_& m)
{
  using namespace ttnte::linalg;

  // =================================================================
  // mm and mv
  m.def("mv", py::overload_cast<const TTEngine&, const TTEngine&>(&mv),
    py::arg("a"), py::arg("b"), py::call_guard<py::gil_scoped_release>());
  m.def("mv",
    py::overload_cast<const TTOperator::Ptr&, const TTState::Ptr&>(&mv),
    py::arg("a"), py::arg("b"), py::call_guard<py::gil_scoped_release>());
  m.def("mm", py::overload_cast<const TTEngine&, const TTEngine&>(&mm),
    py::arg("a"), py::arg("b"), py::call_guard<py::gil_scoped_release>());
  m.def("mm",
    py::overload_cast<const TTOperator::Ptr&, const TTOperator::Ptr&>(&mm),
    py::arg("a"), py::arg("b"), py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // elementwise_divide
  m.def("elementwise_divide",
    py::overload_cast<const TTEngine&, const TTEngine&, int,
      std::optional<TTEngine>, double, int, int, int, int, std::string, int,
      int, bool, std::optional<std::string>>(&elementwise_divide),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 50,
    py::arg("initial_guess") = py::none(), py::arg("eps") = 1e-12,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
    py::arg("trunc_norm") = "res", py::arg("local_iterations") = 40,
    py::arg("resets") = 2, py::arg("verbose") = false,
    py::arg("preconditioner") = py::none(),
    py::call_guard<py::gil_scoped_release>());

  m.def("elementwise_divide",
    py::overload_cast<const TTState::Ptr&, const TTState::Ptr&, int,
      std::optional<TTState::Ptr>, double, int, int, int, int, std::string, int,
      int, bool, std::optional<std::string>>(&elementwise_divide),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 50,
    py::arg("initial_guess") = py::none(), py::arg("eps") = 1e-12,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
    py::arg("trunc_norm") = "res", py::arg("local_iterations") = 40,
    py::arg("resets") = 2, py::arg("verbose") = false,
    py::arg("preconditioner") = py::none(),
    py::call_guard<py::gil_scoped_release>());

  m.def("elementwise_divide",
    py::overload_cast<const TTOperator::Ptr&, const TTOperator::Ptr&, int,
      std::optional<TTOperator::Ptr>, double, int, int, int, int, std::string,
      int, int, bool, std::optional<std::string>>(&elementwise_divide),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 50,
    py::arg("initial_guess") = py::none(), py::arg("eps") = 1e-12,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
    py::arg("trunc_norm") = "res", py::arg("local_iterations") = 40,
    py::arg("resets") = 2, py::arg("verbose") = false,
    py::arg("preconditioner") = py::none(),
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // dmrg_mv
  m.def("dmrg_mv",
    py::overload_cast<const TTEngine&, const TTEngine&, std::optional<TTEngine>,
      int, double, int, int, bool>(&dmrg_mv),
    py::arg("A"), py::arg("x"), py::arg("y0") = py::none(),
    py::arg("nswp") = 20, py::arg("eps") = 1e-12,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());

  m.def("dmrg_mv",
    py::overload_cast<const TTOperator::Ptr&, const TTState::Ptr&,
      std::optional<TTState::Ptr>, int, double, int, int, bool>(&dmrg_mv),
    py::arg("A"), py::arg("x"), py::arg("y0") = py::none(),
    py::arg("nswp") = 20, py::arg("eps") = 1e-12,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // fast_hadamard
  m.def("fast_hadamard",
    py::overload_cast<const TTEngine&, const TTEngine&, double, int>(
      &fast_hadamard),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::call_guard<py::gil_scoped_release>());

  m.def("fast_hadamard",
    py::overload_cast<const TTState::Ptr&, const TTState::Ptr&, double, int>(
      &fast_hadamard),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::call_guard<py::gil_scoped_release>());

  m.def("fast_hadamard",
    py::overload_cast<const TTOperator::Ptr&, const TTOperator::Ptr&, double,
      int>(&fast_hadamard),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // fast_mm
  m.def("fast_mm",
    py::overload_cast<const TTEngine&, const TTEngine&, double, int>(&fast_mm),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::call_guard<py::gil_scoped_release>());

  m.def("fast_mm",
    py::overload_cast<const TTOperator::Ptr&, const TTOperator::Ptr&, double,
      int>(&fast_mm),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // fast_mv
  m.def("fast_mv",
    py::overload_cast<const TTEngine&, const TTEngine&, double, int>(&fast_mv),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::call_guard<py::gil_scoped_release>());

  m.def("fast_mv",
    py::overload_cast<const TTOperator::Ptr&, const TTState::Ptr&, double, int>(
      &fast_mv),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // amen_mm
  m.def("amen_mm",
    py::overload_cast<const TTEngine&, const TTEngine&, int,
      std::optional<TTEngine>, double, int, int, int, bool>(&amen_mm),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 22,
    py::arg("x0") = py::none(), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("kick2") = 0, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());

  m.def("amen_mm",
    py::overload_cast<const TTOperator::Ptr&, const TTOperator::Ptr&, int,
      std::optional<TTOperator::Ptr>, double, int, int, int, bool>(&amen_mm),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 22,
    py::arg("x0") = py::none(), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("kick2") = 0, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // amen_mv
  m.def("amen_mv",
    py::overload_cast<const TTEngine&, const TTEngine&, int,
      std::optional<TTEngine>, double, int, int, int, bool>(&amen_mv),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 22,
    py::arg("x0") = py::none(), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("kick2") = 0, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());

  m.def("amen_mv",
    py::overload_cast<const TTOperator::Ptr&, const TTState::Ptr&, int,
      std::optional<TTState::Ptr>, double, int, int, int, bool>(&amen_mv),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 22,
    py::arg("x0") = py::none(), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("kick2") = 0, py::arg("verbose") = false,
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // amen_solve
  m.def("amen_solve",
    py::overload_cast<const TTEngine&, const TTEngine&, std::optional<TTEngine>,
      int, double, int, int, int, int, int, int, bool, int>(&amen_solve),
    py::arg("A"), py::arg("b"), py::arg("x0") = py::none(),
    py::arg("nswp") = 22, py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
    py::arg("local_iterations") = 40, py::arg("resets") = 2,
    py::arg("verbose") = false, py::arg("preconditioner") = 0,
    py::call_guard<py::gil_scoped_release>());

  m.def("amen_solve",
    py::overload_cast<const TTOperator::Ptr&, const TTState::Ptr&,
      std::optional<TTState::Ptr>, int, double, int, int, int, int, int, int,
      bool, int>(&amen_solve),
    py::arg("A"), py::arg("b"), py::arg("x0") = py::none(),
    py::arg("nswp") = 22, py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
    py::arg("local_iterations") = 40, py::arg("resets") = 2,
    py::arg("verbose") = false, py::arg("preconditioner") = 0,
    py::call_guard<py::gil_scoped_release>());
}
