#include "ttnte/linalg/krylov.hpp"
#include "ttnte/utils/exception.hpp"
#include <barrier>
#include <thread>
#include <torch/extension.h>

namespace py = pybind11;

namespace {

using namespace ttnte::linalg;

using Result = std::tuple<torch::Tensor, int64_t, double, bool>;

/// Validated dense problem: A (n, n) row-major, b and the initial guess.
struct Problem {
  torch::Tensor A;
  std::vector<double> b, x;
  int64_t n = 0;
};

Problem make_problem(const char* where, const torch::Tensor& A,
  const torch::Tensor& b, const std::optional<torch::Tensor>& x0)
{
  if (A.dim() != 2 || A.size(0) != A.size(1))
    throw ttnte::utils::runtime_error(where, "A must be square");
  Problem p;
  p.n = A.size(0);
  if (b.numel() != p.n || (x0.has_value() && x0->numel() != p.n))
    throw ttnte::utils::runtime_error(
      where, "b and x0 must have one entry per row of A");
  p.A = A.to(torch::kFloat64).contiguous();
  auto bd = b.to(torch::kFloat64).contiguous();
  p.b.assign(bd.data_ptr<double>(), bd.data_ptr<double>() + p.n);
  p.x.assign(p.n, 0.0);
  if (x0.has_value()) {
    auto xd = x0->to(torch::kFloat64).contiguous();
    p.x.assign(xd.data_ptr<double>(), xd.data_ptr<double>() + p.n);
  }
  return p;
}

/// out = A[rows r0..r1) in, for the full vector `in`.
void apply_rows(const double* a, int64_t n, int64_t r0, int64_t r1,
  const double* in, double* out)
{
  for (int64_t i = r0; i < r1; i++) {
    double s = 0.0;
    for (int64_t j = 0; j < n; j++)
      s += a[i * n + j] * in[j];
    out[i - r0] = s;
  }
}

torch::Tensor to_tensor(std::vector<double>& x)
{
  return torch::from_blob(
    x.data(), {static_cast<int64_t>(x.size())}, torch::kFloat64)
    .clone();
}

/// gmres() with the vectors split in `parts` contiguous slices, one thread
/// per slice standing in for one rank: the operator gathers the full vector
/// (as a sweep exchanges traces) and `reduce` sums the per-slice partial
/// inner products in a fixed order (as an allreduce). Every thread must
/// reach the same result.
Result distributed_gmres(
  Problem& p, int64_t parts, int64_t restart, double tol, int64_t max_applies)
{
  const char* where = "ttnte::linalg::krylov::dense_gmres";
  if (parts < 1 || parts > p.n)
    throw ttnte::utils::runtime_error(where, "Expected 1 <= parts <= n");
  const int64_t n = p.n;
  const double* a = p.A.data_ptr<double>();
  std::vector<int64_t> bounds(parts + 1);
  for (int64_t q = 0; q <= parts; q++)
    bounds[q] = q * n / parts;

  std::vector<double> full(n), partial(parts);
  std::barrier sync(static_cast<std::ptrdiff_t>(parts));
  std::vector<krylov::KrylovResult> results(parts);
  std::vector<std::vector<double>> xs(parts);
  std::vector<std::thread> threads;
  for (int64_t q = 0; q < parts; q++) {
    threads.emplace_back([&, q] {
      const int64_t r0 = bounds[q], r1 = bounds[q + 1];
      std::vector<double> b(p.b.begin() + r0, p.b.begin() + r1);
      xs[q].assign(p.x.begin() + r0, p.x.begin() + r1);
      auto apply = [&](const double* in, double* out) {
        std::copy(in, in + (r1 - r0), full.begin() + r0);
        sync.arrive_and_wait();
        apply_rows(a, n, r0, r1, full.data(), out);
        sync.arrive_and_wait();
      };
      auto reduce = [&](double local) {
        partial[q] = local;
        sync.arrive_and_wait();
        double s = 0.0;
        for (double v : partial)
          s += v;
        sync.arrive_and_wait();
        return s;
      };
      results[q] =
        krylov::gmres(apply, b, xs[q], restart, tol, max_applies, reduce);
    });
  }
  for (auto& t : threads)
    t.join();
  for (int64_t q = 1; q < parts; q++) {
    if (results[q].num_applies != results[0].num_applies ||
        results[q].converged != results[0].converged ||
        results[q].relative_residual != results[0].relative_residual)
      throw ttnte::utils::runtime_error(
        where, "The slices reached different results");
  }
  std::vector<double> x;
  for (auto& slice : xs)
    x.insert(x.end(), slice.begin(), slice.end());
  return {to_tensor(x), results[0].num_applies, results[0].relative_residual,
    results[0].converged};
}

} // namespace

void register_krylov(py::module_& m)
{
  m.def(
    "dense_gmres",
    [](const torch::Tensor& A, const torch::Tensor& b,
      std::optional<torch::Tensor> x0, int64_t restart, double tol,
      int64_t max_applies, int64_t parts) -> Result {
      auto p = make_problem("ttnte::linalg::krylov::dense_gmres", A, b, x0);
      if (parts != 1)
        return distributed_gmres(p, parts, restart, tol, max_applies);
      const double* a = p.A.data_ptr<double>();
      const int64_t n = p.n;
      auto apply = [a, n](const double* in, double* out) {
        apply_rows(a, n, 0, n, in, out);
      };
      auto res = krylov::gmres(apply, p.b, p.x, restart, tol, max_applies);
      return {
        to_tensor(p.x), res.num_applies, res.relative_residual, res.converged};
    },
    "Restarted GMRES on a dense matrix (testing aid for the matrix-free "
    "krylov::gmres). With parts > 1 the vectors are split over that many "
    "threads, each running gmres() on its slice with an inner-product "
    "reducer, as ranks do. Returns (x, num_applies, relative_residual, "
    "converged).",
    py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
    py::arg("restart") = 30, py::arg("tol") = 1e-10,
    py::arg("max_applies") = 1000, py::arg("parts") = 1,
    py::call_guard<py::gil_scoped_release>());

  m.def(
    "dense_gmres_progress",
    [](const torch::Tensor& A, const torch::Tensor& b,
      std::optional<torch::Tensor> x0, int64_t restart, double tol,
      int64_t max_applies) {
      auto p =
        make_problem("ttnte::linalg::krylov::dense_gmres_progress", A, b, x0);
      const double* a = p.A.data_ptr<double>();
      const int64_t n = p.n;
      int64_t applies = 0;
      std::vector<std::tuple<double, bool, int64_t>> calls;
      auto apply = [&](const double* in, double* out) {
        apply_rows(a, n, 0, n, in, out);
        applies++;
      };
      auto progress = [&](double residual, bool true_residual) {
        calls.emplace_back(residual, true_residual, applies);
      };
      auto res =
        krylov::gmres(apply, p.b, p.x, restart, tol, max_applies, {}, progress);
      return std::make_tuple(to_tensor(p.x), res.num_applies,
        res.relative_residual, res.converged, calls);
    },
    "dense_gmres() recording the progress callback: returns (x, "
    "num_applies, relative_residual, converged, calls), each call a "
    "(residual, true_residual, applications made before it) tuple.",
    py::arg("A"), py::arg("b"), py::arg("x0") = std::nullopt,
    py::arg("restart") = 30, py::arg("tol") = 1e-10,
    py::arg("max_applies") = 1000, py::call_guard<py::gil_scoped_release>());
}
