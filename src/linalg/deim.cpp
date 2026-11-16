// DEIM and Q-DEIM row selection, and the quadrature-weighted orthonormal
// basis of Han et al. (2026)'s angular update.
#include "ttnte/linalg/deim.hpp"
#include "ttnte/linalg/lapack.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <limits>

namespace ttnte::linalg {

namespace {

const auto F64 = torch::TensorOptions().dtype(torch::kFloat64);

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
  const int64_t r = V.size(1);
  std::vector<int64_t> rows {V.select(1, 0).abs().argmax().item<int64_t>()};
  for (int64_t j = 1; j < r; j++) {
    // Interpolate column j on the rows so far, pick the largest residual
    const auto I = torch::tensor(rows, torch::kInt64);
    const auto Vj = V.narrow(1, 0, j);
    const auto P = Vj.index_select(0, I); // (j, j)
    const auto b = V.select(1, j).index_select(0, I).unsqueeze(1);
    if (torch::linalg_matrix_rank(P).item<int64_t>() < j)
      throw utils::runtime_error("ttnte::linalg::deim_select",
        "Singular interpolation matrix (V rank deficient?)");
    const auto c = torch::linalg_solve(P, b); // (j, 1)
    const auto res = V.select(1, j) - Vj.mm(c).squeeze(1);
    rows.push_back(res.abs().argmax().item<int64_t>());
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
  if (L.dim() != 2 || w.dim() != 1 || w.size(0) != L.size(0))
    throw utils::runtime_error(
      "ttnte::linalg::weighted_orthonormal", "L must be (n, r) and w (n)");
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
  if (V.dim() != 2 || static_cast<int64_t>(I.size()) < V.size(1))
    throw utils::runtime_error("ttnte::linalg::lebesgue_constant",
      "V must be (n, r) and I hold at least r rows");
  auto Vw = V.to(torch::kFloat64);
  if (w) {
    // In ||g||_w = ||diag(sqrt w) g||_2 the interpolation projector
    // P = V V(I, :)^{-1} S^T has norm ||Q(I, :)^{-1}||_2 with Q an orthonormal
    // basis of diag(sqrt w) V: the weights of the rows I cancel
    if (w->dim() != 1 || w->size(0) != V.size(0))
      throw utils::runtime_error(
        "ttnte::linalg::lebesgue_constant", "w must be (n)");
    const auto wd = w->to(torch::kFloat64);
    if (!(wd > 0).all().item<bool>())
      throw utils::runtime_error(
        "ttnte::linalg::lebesgue_constant", "Weights must be positive");
    Vw = wd.sqrt().unsqueeze(1) * Vw;
  }
  const auto Vo = std::get<0>(torch::linalg_qr(Vw));
  const auto s =
    torch::linalg_svdvals(Vo.index_select(0, torch::tensor(I, torch::kInt64)));
  const double smin = s.min().item<double>();
  return smin > 0.0 ? 1.0 / smin : std::numeric_limits<double>::infinity();
}

} // namespace ttnte::linalg
