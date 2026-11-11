#include "ttnte/cross/pivots.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_pivots(py::module_& m)
{
  using namespace ttnte::cross;

  m.def("pivot_columns", &pivot_columns, py::arg("R"), py::arg("eps"),
    "Columns of R by Householder QR with column pivoting until the unpicked "
    "part is below eps ||R||_F, in pivot order (testing aid).");
  m.def("direction_sketch", &direction_sketch, py::arg("direction"),
    py::arg("k"), py::arg("n"),
    "(k, n) Gaussian sketch seeded by the direction index (testing aid).");
  m.def(
    "select_block_grams",
    [](const torch::Tensor& grams, const std::vector<int64_t>& sizes,
      const std::vector<double>& cost, std::vector<uint8_t> picked, double cond,
      double near) {
      select_block_grams(grams, sizes, cost, picked, cond, near);
      return std::vector<bool>(picked.begin(), picked.end());
    },
    py::arg("grams"), py::arg("sizes"), py::arg("cost"), py::arg("picked"),
    py::arg("cond") = 1e-2, py::arg("near") = 0.25,
    "Block selection on (nk, r, r) block Grams; returns the chosen flags "
    "(testing aid).");
  m.def(
    "select_single_rows",
    [](const torch::Tensor& rows, const std::vector<double>& cost,
      std::vector<uint8_t> picked, double cond, double near) {
      select_single_rows(rows, cost, picked, cond, near);
      return std::vector<bool>(picked.begin(), picked.end());
    },
    py::arg("rows"), py::arg("cost"), py::arg("picked"), py::arg("cond") = 1e-2,
    py::arg("near") = 0.25,
    "select_block_grams() for one-row blocks by pivoted Gram-Schmidt; "
    "returns the chosen flags (testing aid).");
}
