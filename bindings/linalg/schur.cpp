#include "ttnte/linalg/schur.hpp"
#include "ttnte/utils/exception.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_schur(py::module_& m)
{
  using namespace ttnte::linalg;

  m.def(
    "complex_schur",
    [](const torch::Tensor& a) {
      if (a.dim() != 2 || a.size(0) != a.size(1))
        throw ttnte::utils::runtime_error(
          "ttnte::linalg::complex_schur", "Expected a square (n, n) matrix");
      auto t = a.to(torch::kComplexDouble).contiguous().clone();
      const int64_t n = t.size(0);
      auto q = torch::empty_like(t);
      if (!complex_schur(n,
            reinterpret_cast<std::complex<double>*>(
              t.data_ptr<c10::complex<double>>()),
            reinterpret_cast<std::complex<double>*>(
              q.data_ptr<c10::complex<double>>()))) {
        throw ttnte::utils::runtime_error(
          "ttnte::linalg::complex_schur", "QR iteration did not converge");
      }
      return std::make_tuple(q, t);
    },
    "Complex Schur decomposition A = Q T Q^* of an (n, n) matrix. Returns "
    "(Q, T).",
    py::arg("a"));
}
