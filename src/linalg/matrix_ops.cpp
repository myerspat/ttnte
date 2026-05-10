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

} // namespace ttnte::linalg
