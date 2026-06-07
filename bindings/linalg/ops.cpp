#include "ttnte/linalg/ops.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_ops(py::module_& m)
{
  using namespace ttnte::linalg;

  // =================================================================
  // Exact Matrix & Vector Products
  // =================================================================
  m.def("mm", py::overload_cast<const Operator&, const Operator&>(&mm),
    py::arg("a"), py::arg("b"), "Compute an exact matrix-matrix product.",
    py::call_guard<py::gil_scoped_release>());

  m.def("mv", py::overload_cast<const Operator&, const State&>(&mv),
    py::arg("a"), py::arg("b"), "Compute an exact matrix-vector product.",
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // Element-Wise Operations & Fast Approximations
  // =================================================================
  m.def("elementwise_divide",
    py::overload_cast<const State&, const State&, int, std::optional<State>,
      double, int, int, int, int, std::string, int, int, bool,
      std::optional<std::string>>(&elementwise_divide),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 50,
    py::arg("initial_guess") = py::none(), py::arg("eps") = 1e-12,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
    py::arg("trunc_norm") = "res", py::arg("local_iterations") = 40,
    py::arg("resets") = 2, py::arg("verbose") = false,
    py::arg("preconditioner") = py::none(),
    "Perform an element-wise division with two States using AMEn.",
    py::call_guard<py::gil_scoped_release>());

  m.def("fast_hadamard",
    py::overload_cast<const Operator&, const State&, double, int>(
      &fast_hadamard),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    "Compute the approximate element-wise product rapidly.",
    py::call_guard<py::gil_scoped_release>());

  m.def("fast_mm",
    py::overload_cast<const Operator&, const Operator&, double, int>(&fast_mm),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    "Compute an approximate matrix-matrix product rapidly.",
    py::call_guard<py::gil_scoped_release>());

  m.def("fast_mv",
    py::overload_cast<const Operator&, const State&, double, int>(&fast_mv),
    py::arg("a"), py::arg("b"), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    "Compute an approximate matrix-vector product rapidly.",
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // AMEn & DMRG Solvers / Products
  // =================================================================
  m.def("dmrg_mv",
    py::overload_cast<const Operator&, const State&, std::optional<State>, int,
      double, int, int, bool>(&dmrg_mv),
    py::arg("A"), py::arg("x"), py::arg("y0") = py::none(),
    py::arg("nswp") = 20, py::arg("eps") = 1e-12,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("verbose") = false,
    "Compute a matrix-vector product using DMRG.",
    py::call_guard<py::gil_scoped_release>());

  m.def("amen_mm",
    py::overload_cast<const Operator&, const Operator&, int,
      std::optional<Operator>, double, int, int, int, bool>(&amen_mm),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 22,
    py::arg("x0") = py::none(), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("kick2") = 0, py::arg("verbose") = false,
    "Compute the matrix-matrix product using AMEn.",
    py::call_guard<py::gil_scoped_release>());

  m.def("amen_mv",
    py::overload_cast<const Operator&, const State&, int, std::optional<State>,
      double, int, int, int, bool>(&amen_mv),
    py::arg("a"), py::arg("b"), py::arg("nswp") = 22,
    py::arg("x0") = py::none(), py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("kickrank") = 4, py::arg("kick2") = 0, py::arg("verbose") = false,
    "Compute the matrix-vector product using AMEn.",
    py::call_guard<py::gil_scoped_release>());

  m.def("amen_solve",
    py::overload_cast<const Operator&, const State&, std::optional<State>, int,
      double, int, int, int, int, int, int, bool, int>(&amen_solve),
    py::arg("A"), py::arg("b"), py::arg("x0") = py::none(),
    py::arg("nswp") = 22, py::arg("eps") = 1e-10,
    py::arg("max_rank") = std::numeric_limits<int>::max(),
    py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
    py::arg("local_iterations") = 40, py::arg("resets") = 2,
    py::arg("verbose") = false, py::arg("preconditioner") = 0,
    "Solve a linear system A @ x = b using AMEn.",
    py::call_guard<py::gil_scoped_release>());

  // =================================================================
  // Function Interpolation & Cross Approximations
  // =================================================================

  // Overload 1: Univariate function_interpolate
  m.def("function_interpolate",
    py::overload_cast<const std::function<torch::Tensor(const torch::Tensor&)>&,
      const State&, double, std::optional<State>, int, int, int, bool>(
      &function_interpolate),
    py::arg("func"), py::arg("x"), py::arg("eps") = 1e-9,
    py::arg("start_tens") = py::none(), py::arg("nswp") = 20,
    py::arg("kick") = 2, py::arg("rmax") = std::numeric_limits<int>::max(),
    py::arg("verbose") = false,
    "Interpolate a State representation for a univariate function using "
    "TT-cross.",
    py::call_guard<py::gil_scoped_release>());

  // Overload 2: Multivariate function_interpolate
  m.def("function_interpolate",
    py::overload_cast<
      const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>&,
      const std::vector<State>&, double, std::optional<State>, int, int, int,
      bool>(&function_interpolate),
    py::arg("func"), py::arg("xs"), py::arg("eps") = 1e-9,
    py::arg("start_tens") = py::none(), py::arg("nswp") = 20,
    py::arg("kick") = 2, py::arg("rmax") = std::numeric_limits<int>::max(),
    py::arg("verbose") = false,
    "Interpolate a State representation for a multivariate function using "
    "TT-cross.",
    py::call_guard<py::gil_scoped_release>());

  // dmrg_cross
  m.def("dmrg_cross",
    py::overload_cast<const std::function<torch::Tensor(const torch::Tensor&)>&,
      const std::vector<int64_t>&, double, int, std::optional<State>, int, int,
      bool, const torch::Device&, const torch::ScalarType&>(&dmrg_cross),
    py::arg("func"), py::arg("N"), py::arg("eps") = 1e-9, py::arg("nswp") = 20,
    py::arg("x0") = py::none(), py::arg("kick") = 2,
    py::arg("rmax") = std::numeric_limits<int>::max(),
    py::arg("verbose") = false, py::arg("device") = torch::kCPU,
    py::arg("dtype") = torch::kFloat64,
    "Interpolate a State representation with index inputs using DMRG-cross.",
    py::call_guard<py::gil_scoped_release>());
}
