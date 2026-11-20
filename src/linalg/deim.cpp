// DEIM and Q-DEIM row selection, and the quadrature-weighted orthonormal
// basis of Han et al. (2026)'s angular update.
#include "ttnte/linalg/deim.hpp"
#include "ttnte/linalg/lapack.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ttnte::linalg {

namespace {

/// Rounding floor of a basis of n rows and r columns: residuals and singular
/// values at most this times the scale are numerically zero
double rounding_floor(int64_t n, int64_t r)
{
  return static_cast<double>(std::max(n, r)) *
         std::numeric_limits<double>::epsilon();
}

void check_basis(const torch::Tensor& V, const char* where)
{
  if (V.dim() != 2 || V.size(1) < 1 || V.size(1) > V.size(0))
    throw utils::runtime_error(where, "V must be (n, r) with 1 <= r <= n");
}

} // namespace

std::vector<int64_t> deim_select(const torch::Tensor& V_in)
{
  check_basis(V_in, "ttnte::linalg::deim_select");
  const auto V = V_in.to(torch::kFloat64).contiguous();
  const int64_t n = V.size(0), r = V.size(1);
  const double floor = rounding_floor(n, r);
  // Each column's pick: the largest residual off the rows chosen so far (a
  // chosen row's residual is zero in exact arithmetic, rounding otherwise:
  // masked so it cannot be picked twice). A largest residual at rounding
  // level means column j is (numerically) in the span of the previous ones
  // at every row: V is rank deficient and V(I, :) would be singular
  std::vector<int64_t> rows;
  for (int64_t j = 0; j < r; j++) {
    auto res = V.select(1, j).clone();
    if (j > 0) {
      const auto I = torch::tensor(rows, torch::kInt64);
      const auto Vj = V.narrow(1, 0, j);
      const auto c = torch::linalg_solve(Vj.index_select(0, I),
        V.select(1, j).index_select(0, I).unsqueeze(1)); // (j, 1)
      res -= Vj.mm(c).squeeze(1);
      res.index_fill_(0, I, 0.0);
    }
    const auto [top, at] = res.abs().max(0);
    const double scale = V.select(1, j).norm().item<double>();
    if (!(top.item<double>() > floor * scale))
      throw utils::runtime_error("ttnte::linalg::deim_select",
        "Column " + std::to_string(j) +
          " is in the span of the previous ones (V rank deficient)");
    rows.push_back(at.item<int64_t>());
  }
  return rows;
}

std::vector<int64_t> qdeim_select(const torch::Tensor& V_in)
{
  check_basis(V_in, "ttnte::linalg::qdeim_select");
  // V (n, r) row-major is V^T (r, n) column-major with leading dimension r
  auto A = V_in.to(torch::kFloat64).contiguous().clone();
  const int m = static_cast<int>(A.size(1)), n = static_cast<int>(A.size(0));
  std::vector<int> jpvt(static_cast<size_t>(n), 0); // all columns free
  std::vector<double> tau(static_cast<size_t>(std::min(m, n)));
  lapack::SingleThreaded single;
  int info = 0, lwork = -1;
  double query = 0.0;
  dgeqp3_(&m, &n, A.data_ptr<double>(), &m, jpvt.data(), tau.data(), &query,
    &lwork, &info);
  if (info != 0)
    throw utils::runtime_error("ttnte::linalg::qdeim_select",
      "dgeqp3 workspace query failed (info " + std::to_string(info) + ")");
  lwork = std::max(1, static_cast<int>(query));
  std::vector<double> work(static_cast<size_t>(lwork));
  dgeqp3_(&m, &n, A.data_ptr<double>(), &m, jpvt.data(), tau.data(),
    work.data(), &lwork, &info);
  if (info != 0)
    throw utils::runtime_error("ttnte::linalg::qdeim_select",
      "dgeqp3 failed (info " + std::to_string(info) + ")");
  std::vector<int64_t> rows(static_cast<size_t>(m));
  for (int k = 0; k < m; k++)
    rows[k] = jpvt[k] - 1; // 1-based
  return rows;
}

std::pair<torch::Tensor, torch::Tensor> weighted_orthonormal(
  const torch::Tensor& L, const torch::Tensor& w)
{
  if (L.dim() != 2 || w.dim() != 1 || w.size(0) != L.size(0) ||
      L.size(1) > L.size(0))
    throw utils::runtime_error("ttnte::linalg::weighted_orthonormal",
      "L must be (n, r) with r <= n and w (n)");
  const auto wd = w.to(torch::kFloat64);
  if (!(wd > 0).all().item<bool>())
    throw utils::runtime_error(
      "ttnte::linalg::weighted_orthonormal", "Weights must be positive");
  // diag(sqrt w) L = Q R, so L = (diag(1 / sqrt w) Q) R with the first
  // factor orthonormal in the w-weighted inner product
  const auto s = wd.sqrt().unsqueeze(1);
  const auto [Q, R] = torch::linalg_qr(s * L.to(torch::kFloat64), "reduced");
  return {(Q / s).contiguous(), R.contiguous()};
}

double lebesgue_constant(const torch::Tensor& V, const std::vector<int64_t>& I,
  const std::optional<torch::Tensor>& w)
{
  if (V.dim() != 2 || V.size(1) > V.size(0) ||
      static_cast<int64_t>(I.size()) != V.size(1))
    throw utils::runtime_error("ttnte::linalg::lebesgue_constant",
      "V must be (n, r) with r <= n and I hold r rows");
  auto Vw = V.to(torch::kFloat64);
  if (w) {
    // In ||g||_w = ||diag(sqrt w) g||_2 the interpolation projector
    // P = V V(I, :)^{-1} S^T has norm ||Q(I, :)^{-1}||_2 with Q an orthonormal
    // basis of diag(sqrt w) V: with |I| = r the weights of the rows I cancel
    // (V(I, :) square)
    if (w->dim() != 1 || w->size(0) != V.size(0))
      throw utils::runtime_error(
        "ttnte::linalg::lebesgue_constant", "w must be (n)");
    const auto wd = w->to(torch::kFloat64);
    if (!(wd > 0).all().item<bool>())
      throw utils::runtime_error(
        "ttnte::linalg::lebesgue_constant", "Weights must be positive");
    Vw = wd.sqrt().unsqueeze(1) * Vw;
  }
  // An orthonormal basis of span(V) from its SVD: a rank-deficient V has no
  // r-dimensional span (QR would complete it with arbitrary columns), so its
  // interpolation is unbounded
  const auto [Vo, sv, Vh] = torch::linalg_svd(Vw, /*full_matrices=*/false);
  if (!(sv.min().item<double>() >
        rounding_floor(V.size(0), V.size(1)) * sv.max().item<double>()))
    return std::numeric_limits<double>::infinity();
  const auto s =
    torch::linalg_svdvals(Vo.index_select(0, torch::tensor(I, torch::kInt64)));
  const double smin = s.min().item<double>();
  return smin > 0.0 ? 1.0 / smin : std::numeric_limits<double>::infinity();
}

} // namespace ttnte::linalg
