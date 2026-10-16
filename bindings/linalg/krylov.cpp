#include "ttnte/linalg/krylov.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_krylov(py::module_& m)
{
  using namespace ttnte::linalg;

  m.def(
    "dense_gmres",
    [](const torch::Tensor& A, const torch::Tensor& b,
      std::optional<torch::Tensor> x0, int64_t restart, double tol,
      int64_t max_applies) {
      auto Ad = A.to(torch::kFloat64).contiguous();
      const int64_t n = Ad.size(0);
      const double* a = Ad.data_ptr<double>();
      auto bd = b.to(torch::kFloat64).contiguous();
      std::vector<double> rhs(bd.data_ptr<double>(), bd.data_ptr<double>() + n);
      std::vector<double> x(n, 0.0);
      if (x0.has_value()) {
        auto xd = x0->to(torch::kFloat64).contiguous();
        x.assign(xd.data_ptr<double>(), xd.data_ptr<double>() + n);
      }
      auto apply = [a, n](const double* in, double* out) {
        for (int64_t i = 0; i < n; i++) {
          double s = 0.0;
          for (int64_t j = 0; j < n; j++)
            s += a[i * n + j] * in[j];
          out[i] = s;
        }
      };
      auto res = krylov::gmres(apply, rhs, x, restart, tol, max_applies);
      auto xt = torch::from_blob(x.data(), {n}, torch::kFloat64).clone();
      return std::make_tuple(
        xt, res.num_applies, res.relative_residual, res.converged);
    },
    "Restarted GMRES on a dense matrix (testing aid for the matrix-free "
    "krylov::gmres). Returns (x, num_applies, relative_residual, "
    "converged).",
    py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
    py::arg("restart") = 30, py::arg("tol") = 1e-10,
    py::arg("max_applies") = 1000, py::call_guard<py::gil_scoped_release>());
}
