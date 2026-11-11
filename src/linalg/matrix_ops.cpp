#include "ttnte/linalg/matrix_ops.hpp"
#include "ttnte/utils/exception.hpp"

#include <ATen/Parallel.h>
#include <algorithm>

namespace ttnte::linalg {

namespace {

/// @brief Rank of a truncation with Frobenius error at most delta.
/// @param sq (n) squared singular values, descending.
/// @return The rank (at least 1, at most max_rank) and the unused error
/// budget sqrt(max(delta^2 - discarded energy, 0)).
std::pair<int64_t, double> truncation_rank(
  const torch::Tensor& sq, double delta, int64_t max_rank)
{
  const int64_t n = sq.size(0);
  // On the host in double: the loop below reads the values directly
  auto tail = torch::cumsum(sq.flip(0), 0) // tail[i]: the last i + 1
                .to(torch::kCPU, torch::kFloat64)
                .contiguous();
  const double* t = tail.data_ptr<double>();
  const double delta_sq = delta * delta;
  int64_t removed = 0;
  while (removed < n && t[removed] <= delta_sq)
    removed++;
  const int64_t rank =
    std::min(std::max<int64_t>(n - removed, 1), std::max<int64_t>(max_rank, 1));
  const double discarded = rank < n ? t[n - rank - 1] : 0.0;
  return {rank, std::sqrt(std::max(delta_sq - discarded, 0.0))};
}

/// Below this sigma_r / sigma_1 the Gram SVD's long factor is
/// re-orthogonalized (its orthogonality error grows as eps (sigma_1 /
/// sigma_r)^2, so about 1e-10 here).
constexpr double GRAM_REORTHOGONALIZE = 1e-3;

} // namespace

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
  if (A.size(0) > SKINNY_RATIO * A.size(1)) {
    // Tall and skinny matrix
    auto [Q, R] = torch::linalg_qr(A, "reduced");
    std::tie(U, S, Vh) = torch::linalg_svd(R);
    U = torch::mm(Q, U);

  } else if (A.size(1) > SKINNY_RATIO * A.size(0)) {
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

  auto sq = S.square();
  if (with_normalize)
    delta *= std::sqrt(sq.sum().item<double>());
  const auto [rank, rest] = truncation_rank(sq, delta, max_rank);
  return {U.slice(1, 0, rank), S.slice(0, 0, rank), Vh.slice(0, 0, rank), delta,
    rest};
}

torch::Tensor blocked_gram(const torch::Tensor& A)
{
  if (A.ndimension() != 2)
    throw utils::runtime_error(
      "ttnte::linalg::blocked_gram", "`A` must be a 2-D tensor");
  // Long dimension of A: rows (tall, A^T A) or columns (wide, A A^T)
  const bool tall = A.size(0) >= A.size(1);
  const int64_t len = tall ? A.size(0) : A.size(1);
  const int64_t r = tall ? A.size(1) : A.size(0);
  auto X = A.contiguous();
  // About eight blocks per thread (fewer blocks made the batched product
  // pick a slow path), none shorter than 64 rows
  const int64_t nb =
    std::clamp<int64_t>(8 * static_cast<int64_t>(at::get_num_threads()), 1,
      std::max<int64_t>(len / 64, 1));
  const int64_t mb = len / nb;
  torch::Tensor main;
  if (tall) {
    auto Y = X.narrow(0, 0, nb * mb).view({nb, mb, r}); // (nb, mb, r)
    main = torch::bmm(Y.transpose(1, 2), Y).sum(0);
  } else {
    // Column blocks as a strided view (no copy): (nb, r, mb)
    auto Y = X.narrow(1, 0, nb * mb).view({r, nb, mb}).permute({1, 0, 2});
    main = torch::bmm(Y, Y.transpose(1, 2)).sum(0);
  }
  if (nb * mb < len) {
    auto rest = X.narrow(tall ? 0 : 1, nb * mb, len - nb * mb);
    main += tall ? rest.t().mm(rest) : rest.mm(rest.t());
  }
  return main;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double>
gram_truncated_svd(const torch::Tensor& A, double delta, int64_t max_rank,
  bool with_normalize, double gram_floor, bool orthonormal)
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
  const bool tall = m > SKINNY_RATIO * n, wide = n > SKINNY_RATIO * m;
  if (!tall && !wide)
    return truncated_svd(A, delta, max_rank, with_normalize);
  // The Gram matrix first: ||A||_F^2 is its trace, which saves a pass over A
  auto G = blocked_gram(A);
  const double norm = std::sqrt(std::max(G.trace().item<double>(), 0.0));
  const double threshold = with_normalize ? delta * norm : delta;
  if (norm == 0.0 || threshold < gram_floor * norm)
    return truncated_svd(A, delta, max_rank, with_normalize);

  // Eigenpairs of the small Gram matrix, largest first
  auto [lambda, W] = torch::linalg_eigh(G);
  lambda = lambda.flip(0).clamp_min(0.0);
  W = W.flip(1);

  // Truncate on the tail energy, as truncated_svd()
  const auto [rank, rest] =
    truncation_rank(lambda.contiguous(), threshold, max_rank);
  auto S = lambda.slice(0, 0, rank).sqrt();
  auto Wr = W.slice(1, 0, rank);
  auto inv = 1.0 / S.clamp_min(1e-300);

  // The 1 / sigma scaling goes on the small matrix, so the long factor is
  // formed in one pass
  torch::Tensor U, Vh;
  if (tall) {
    Vh = Wr.t().contiguous();
    U = A.mm(Wr * inv.unsqueeze(0));
  } else {
    U = Wr.contiguous();
    Vh = (Wr.t() * inv.unsqueeze(1)).mm(A);
  }

  // Ill-conditioned: one CholeskyQR pass on the long factor L = Q R, and
  // the small R S (or S R^T) refactored, restores orthogonality
  const double s0 = S[0].item<double>();
  const double sr = S[rank - 1].item<double>();
  if (orthonormal && sr < GRAM_REORTHOGONALIZE * s0) {
    auto& L = tall ? U : Vh;
    auto [R, info] =
      torch::linalg_cholesky_ex(blocked_gram(L), /*upper=*/true); // R^T R
    if (info.item<int64_t>() != 0)
      return truncated_svd(A, delta, max_rank, with_normalize);
    auto Rinv = torch::linalg_solve_triangular(
      R, torch::eye(rank, R.options()), /*upper=*/true, /*left=*/true);
    if (tall) {
      // A = (U R^{-1}) (R S Vh)
      auto [U2, S2, Vh2] = torch::linalg_svd(R * S.unsqueeze(0), false);
      U = U.mm(Rinv.mm(U2));
      S = S2;
      Vh = Vh2.mm(Vh);
    } else {
      // A = (U S R^T) (R^{-T} Vh)
      auto [U2, S2, Vh2] = torch::linalg_svd(S.unsqueeze(1) * R.t(), false);
      U = U.mm(U2);
      S = S2;
      Vh = Vh2.mm(Rinv.t().mm(Vh));
    }
  }
  return {U, S, Vh, threshold, rest};
}

} // namespace ttnte::linalg
