#include "ttnte/linalg/matrix_ops.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::linalg {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double>
truncated_svd(
  const torch::Tensor& A, double delta, int64_t max_rank, bool with_normalize)
{
  // Perform checks
  if (A.ndimension() != 2) {
    throw utils::runtime_error(
      "ttnte::linalg::truncated_svd", "`A` must be a 2-D tensor");
  } else if (delta < 0 || max_rank < 1) {
    throw utils::runtime_error("ttnte::linalg::truncated_svd",
      "`delta` must be greater than or equal to zero and `max_rank` must be\n"
      "greater than or equal to one");
  }

  torch::Tensor U, S, Vh;

  // Compute safe SVD
  if (A.size(0) > 10 * A.size(1)) {
    // Tall and skinny matrix
    auto [Q, R] = torch::linalg_qr(A, "reduced");
    std::tie(U, S, Vh) = torch::linalg_svd(R);
    U = torch::mm(Q, U);

  } else if (A.size(1) > 10 * A.size(0)) {
    // Short and wide matrix: A = R^T Q^T from the QR of A^T
    auto [Q, R] = torch::linalg_qr(A.t(), "reduced");
    std::tie(U, S, Vh) = torch::linalg_svd(R.t());
    Vh = torch::mm(Vh, Q.t());

  } else {
    // Normal matrices
    try {
      std::tie(U, S, Vh) = torch::linalg_svd(A, false);

    } catch (const c10::Error& e) {
      auto [Q, R] = torch::linalg_qr(A, "reduced");
      std::tie(U, S, Vh) = torch::linalg_svd(R);
      U = torch::mm(Q, U);
    }
  }

  // Compute the squares of the singular values
  auto eig_cumsum = torch::flip(S, {0});
  eig_cumsum.pow_(2);

  // Normalize delta
  if (with_normalize) {
    delta *= torch::sqrt(eig_cumsum.sum()).item<double>();
  }

  // Compute cumulative sum
  eig_cumsum = torch::cumsum(eig_cumsum, 0);

  double delta_sq = delta * delta;
  int64_t removed_rank = torch::sum(eig_cumsum <= delta_sq).item<int64_t>();
  int64_t trunc_rank = std::min(
    std::max(S.size(0) - removed_rank, static_cast<int64_t>(1)), max_rank);

  return {U.slice(1, 0, trunc_rank), S.slice(0, 0, trunc_rank),
    Vh.slice(0, 0, trunc_rank), delta,
    (removed_rank > 0)
      ? std::sqrt(
          std::max(delta_sq - eig_cumsum[removed_rank - 1].item<double>(),
            static_cast<double>(0.0)))
      : delta};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double>
gram_truncated_svd(const torch::Tensor& A, double delta, int64_t max_rank,
  bool with_normalize, double gram_floor)
{
  if (A.ndimension() != 2) {
    throw utils::runtime_error(
      "ttnte::linalg::gram_truncated_svd", "`A` must be a 2-D tensor");
  } else if (delta < 0 || max_rank < 1) {
    throw utils::runtime_error("ttnte::linalg::gram_truncated_svd",
      "`delta` must be greater than or equal to zero and `max_rank` must be\n"
      "greater than or equal to one");
  }
  const int64_t m = A.size(0), n = A.size(1);
  const bool tall = m > 10 * n, wide = n > 10 * m;
  const double norm = A.norm().item<double>();
  const double threshold = with_normalize ? delta * norm : delta;
  if ((!tall && !wide) || norm == 0.0 || threshold < gram_floor * norm)
    return truncated_svd(A, delta, max_rank, with_normalize);

  // Eigenpairs of the small Gram matrix, largest first
  auto G = tall ? A.t().mm(A) : A.mm(A.t());
  auto [lambda, W] = torch::linalg_eigh(G);
  lambda = lambda.flip(0).clamp_min(0.0);
  W = W.flip(1);

  // Truncate on the tail energy, as truncated_svd()
  auto tail = torch::cumsum(lambda.flip(0), 0);
  const double delta_sq = threshold * threshold;
  const int64_t removed = torch::sum(tail <= delta_sq).item<int64_t>();
  const int64_t rank = std::min(
    std::max(lambda.size(0) - removed, static_cast<int64_t>(1)), max_rank);
  auto S = lambda.slice(0, 0, rank).sqrt();
  auto Wr = W.slice(1, 0, rank);
  auto inv = 1.0 / S.clamp_min(1e-300);

  torch::Tensor U, Vh;
  if (tall) {
    Vh = Wr.t().contiguous();
    U = A.mm(Wr) * inv.unsqueeze(0);
  } else {
    U = Wr.contiguous();
    Vh = Wr.t().mm(A) * inv.unsqueeze(1);
  }
  const double rest =
    removed > 0
      ? std::sqrt(std::max(delta_sq - tail[removed - 1].item<double>(), 0.0))
      : threshold;
  return {U, S, Vh, threshold, rest};
}

} // namespace ttnte::linalg
