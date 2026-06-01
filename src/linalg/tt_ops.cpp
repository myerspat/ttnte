#include "ttnte/linalg/tt_ops.hpp"
#include "ttnte/linalg/matrix_ops.hpp"
#include "ttnte/python/torchtt.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::linalg::torchtt {
#include "amen_solve.h"
#include "dmrg_mv.h"
} // namespace ttnte::linalg::torchtt

namespace {

std::pair<torch::Tensor, torch::Tensor> swap_cores(const torch::Tensor& core_a,
  const torch::Tensor& core_b, double eps, int max_rank)
{
  // Get the shapes of each core
  int64_t a = core_a.size(0);
  int64_t b = core_a.size(1);
  int64_t c = core_a.size(2);
  int64_t e = core_b.size(1);
  int64_t f = core_b.size(2);
  int64_t g = core_b.size(3);

  // Merge to create a supercore
  torch::Tensor supercore = torch::einsum("abcd,defg->aefbcg", {core_a, core_b})
                              .reshape({a * e * f, b * c * g});

  // Run truncated SVD
  auto [u, s, vh, _1, _2] =
    ttnte::linalg::truncated_svd(supercore, eps, max_rank, true);

  return {(u * s).reshape({a, e, f, -1}), vh.reshape({-1, b, c, g})};
}

} // namespace

namespace ttnte::linalg {

TTEngine mm(const TTEngine& a, const TTEngine& b)
{
  // Get the cores
  const auto& a_cores = a.get_cores();
  const auto& b_cores = b.get_cores();

  // Checks
  if (a_cores.size() != b_cores.size()) {
    throw utils::runtime_error(
      "ttnte::linalg::mm", "`a` and `b` must have the same number of TT-cores");
  }
  if (a.get_device() != b.get_device()) {
    throw utils::runtime_error(
      "ttnte::linalg::mm", "`a` and `b` must be on the same device");
  }
  size_t num_cores = a_cores.size();

  // Contract free indices
  TTEngine::Tensors cores;
  cores.reserve(num_cores);
  for (size_t i = 0; i < num_cores; i++) {
    const auto& a_core = a_cores[i];
    const auto& b_core = b_cores[i];

    // Shapes of each
    const auto& a_shape = a_core.sizes();
    const auto& b_shape = b_core.sizes();

    // Check sizes
    if (a_shape[2] != b_shape[1]) {
      throw utils::runtime_error("ttnte::linalg::mm",
        "The `n_modes` of `a` must equal the `m_modes` of `b`");
    }

    // Perform contraction and reshape
    cores.push_back(torch::einsum("abcd,ecfg->aebfdg", {a_core, b_core})
        .reshape({a_shape[0] * b_shape[0], a_shape[1], b_shape[2],
          a_shape[3] * b_shape[3]}));
  }

  return TTEngine(cores, false);
}

TTEngine elementwise_divide(const TTEngine& a, const TTEngine& b, int nswp,
  std::optional<TTEngine> initial_guess, double eps, int max_rank, int max_full,
  int kickrank, int kick2, std::string trunc_norm, int local_iterations,
  int resets, bool verbose, std::optional<std::string> preconditioner)
{
  return python::torchtt::Acquire().amen_divide(a, b, nswp, initial_guess, eps,
    max_rank, max_full, kickrank, kick2, trunc_norm, local_iterations, resets,
    verbose, preconditioner);
}

TTEngine dmrg_mv(const TTEngine& A, const TTEngine& x,
  std::optional<TTEngine> y0, int nswp, double eps, int max_rank, int kickrank,
  bool verbose)
{
  // Get cores for each portion
  std::vector<torch::Tensor> A_cores(
    A.get_cores().begin(), A.get_cores().end());
  std::vector<torch::Tensor> x_cores(
    x.get_cores().begin(), x.get_cores().end());
  std::vector<torch::Tensor> y0_cores =
    y0.has_value() ? std::vector<torch::Tensor>(
                       y0->get_cores().begin(), y0->get_cores().end())
                   : std::vector<torch::Tensor>();

  bool given_y0 = !y0_cores.empty();
  if (A_cores.size() != x_cores.size() ||
      (x_cores.size() != y0_cores.size() && given_y0)) {
    throw utils::runtime_error("ttnte::linalg::dmrg_mv",
      "`A` and `x` must have the same number of cores and `y` must either be\n"
      "empty or have the same number of cores");
  }

  // Vector of sizes
  std::vector<int64_t> M;
  std::vector<int64_t> N;
  std::vector<int64_t> rx;
  std::vector<int64_t> ry0;

  int64_t ndim = A_cores.size();
  N.reserve(ndim);
  M.reserve(ndim);
  rx.reserve(ndim + 1);
  ry0.reserve(ndim + 1);

  // Squeeze the cores of the vector
  for (size_t i = 0; i < ndim; i++) {
    const auto& A_core = A_cores[i];
    auto& x_core = x_cores[i];

    M.push_back(A_core.size(1));
    N.push_back(A_core.size(2));
    rx.push_back(x_core.size(0));

    if (N.back() != x_core.size(1)) {
      throw utils::runtime_error("ttnte::linalg::dmrg_mv",
        "The `n_modes` of `A` must equal the `m_modes` of `x`");
    }
    if (x_core.size(2) != 1) {
      throw utils::runtime_error("ttnte::linalg::dmrg_mv",
        "`x` must be a TT-vector with all `m_modes` equal to 1");
    }
    x_core = x_core.squeeze(2);

    if (given_y0) {
      auto& y0_core = y0_cores[i];
      ry0.push_back(y0_cores[i].size(0));

      if (M.back() != y0_core.size(1)) {
        throw utils::runtime_error("ttnte::linalg::dmrg_mv",
          "The `n_modes` of `A` must equal the `n_modes` of `y`");
      }
      if (y0_core.size(2) != 1) {
        throw utils::runtime_error("ttnte::linalg::dmrg_mv",
          "`y` must be a TT-vector with all `m_modes` equal to 1");
      }
      y0_core = y0_core.squeeze(2);
    }
  }
  rx.push_back(1);
  if (given_y0) {
    ry0.push_back(1);
  }

  // Run torchtt DMRG matrix-vector product algorithm
  auto y_cores = torchtt::dmrg_mv(A_cores, x_cores, y0_cores, M, N, rx, ry0,
    nswp, eps, max_rank, kickrank, verbose);

  return TTEngine(TTEngine::Tensors(y_cores.cbegin(), y_cores.cend()), true);
}

TTEngine fast_hadamard(
  const TTEngine& a, const TTEngine& b, double eps, int max_rank)
{
  const auto& cores_a = a.get_cores();
  const auto& cores_b = b.get_cores();

  // Checks
  if (cores_a.size() != cores_b.size()) {
    throw utils::runtime_error("ttnte::linalg::fast_hadamard",
      "`a` and `b` must have the same number of cores");
  }
  if (a.get_device() != b.get_device()) {
    throw utils::runtime_error(
      "ttnte::linalg::fast_hadamard", "`a` and `b` must be on the same device");
  }
  for (size_t i = 0; i < cores_a.size(); i++) {
    const auto& core_a = cores_a[i];
    const auto& core_b = cores_b[i];

    if (core_a.size(1) != core_b.size(1) || core_a.size(2) != core_b.size(2)) {
      throw utils::runtime_error("ttnte::linalg::fast_hadamard",
        "The free indices of `a` and `b` must match in size and order");
    }
  }
  size_t num_cores = cores_a.size();

  // Flip the b TT
  TTEngine::Tensors cores_b_flip;
  cores_b_flip.reserve(num_cores);
  for (auto it = cores_b.rbegin(); it != cores_b.rend(); it++) {
    cores_b_flip.push_back(it->permute({3, 1, 2, 0}).contiguous());
  }

  // Iterate through the cores
  for (size_t i = 0; i < num_cores; i++) {
    const auto& core_a = cores_a[num_cores - 1 - i];
    auto& core_b = cores_b_flip[0];

    // Contract
    core_b = at::einsum("abcd,dbce->abce", {core_a, core_b});

    if (i != num_cores - 1) {
      for (int64_t j = i; j >= 0; j--) {
        auto [c_left, c_right] =
          swap_cores(cores_b_flip[j], cores_b_flip[j + 1], eps, max_rank);
        cores_b_flip[j] = c_left;
        cores_b_flip[j + 1] = c_right;
      }
    }
  }

  return TTEngine(cores_b_flip, false);
}

TTEngine fast_mm(const TTEngine& a, const TTEngine& b, double eps, int max_rank)
{
  const auto& cores_a = a.get_cores();
  const auto& cores_b = b.get_cores();

  // Checks
  if (cores_a.size() != cores_b.size()) {
    throw utils::runtime_error("ttnte::linalg::fast_mm",
      "`a` and `b` must have the same number of cores");
  }
  if (a.get_device() != b.get_device()) {
    throw utils::runtime_error(
      "ttnte::linalg::tt::fast_mm", "`a` and `b` must be on the same device");
  }
  for (size_t i = 0; i < cores_a.size(); i++) {
    const auto& core_a = cores_a[i];
    const auto& core_b = cores_b[i];

    if (core_a.size(2) != core_b.size(1)) {
      throw utils::runtime_error("ttnte::linalg::fast_mm",
        "The `n_modes` of `a` must equal the `m_modes` of `b`");
    }
  }
  size_t num_cores = cores_a.size();

  // Flip the b TT
  TTEngine::Tensors cores_b_flip;
  cores_b_flip.reserve(num_cores);
  for (auto it = cores_b.rbegin(); it != cores_b.rend(); it++) {
    cores_b_flip.push_back(it->permute({3, 1, 2, 0}).contiguous());
  }

  // Iterate through the cores
  for (size_t i = 0; i < num_cores; i++) {
    const auto& core_a = cores_a[num_cores - 1 - i];
    auto& core_b = cores_b_flip[0];

    // Contract
    core_b = at::einsum("abcd,dcef->abef", {core_a, core_b});

    if (i != num_cores - 1) {
      for (int64_t j = i; j >= 0; j--) {
        auto [c_left, c_right] =
          swap_cores(cores_b_flip[j], cores_b_flip[j + 1], eps, max_rank);
        cores_b_flip[j] = c_left;
        cores_b_flip[j + 1] = c_right;
      }
    }
  }

  return TTEngine(cores_b_flip, false);
}

TTEngine amen_mm(const linalg::TTEngine& a, const linalg::TTEngine& b, int nswp,
  std::optional<linalg::TTEngine> x0, double eps, int max_rank, int kickrank,
  int kick2, bool verbose)
{
  return python::torchtt::Acquire().amen_mm(
    a, b, nswp, x0, eps, max_rank, kickrank, kick2, verbose);
}

TTEngine amen_solve(const linalg::TTEngine& A, const linalg::TTEngine& b,
  std::optional<linalg::TTEngine> x0, int nswp, double eps, int max_rank,
  int max_full, int kickrank, int kick2, int local_iterations, int resets,
  bool verbose, int preconditioner)
{
  // Get cores for each portion
  std::vector<torch::Tensor> A_cores(
    A.get_cores().begin(), A.get_cores().end());
  std::vector<torch::Tensor> b_cores(
    b.get_cores().begin(), b.get_cores().end());
  std::vector<torch::Tensor> x0_cores =
    x0.has_value() ? std::vector<torch::Tensor>(
                       x0->get_cores().begin(), x0->get_cores().end())
                   : std::vector<torch::Tensor>();

  bool given_x0 = !x0_cores.empty();
  if (A_cores.size() != b_cores.size() ||
      (b_cores.size() != x0_cores.size() && given_x0)) {
    throw utils::runtime_error("ttnte::linalg::amen_solve",
      "`A` and `b` must have the same number of cores and `x0` must either be\n"
      "empty or have the same number of cores");
  }

  // Vector of sizes
  std::vector<uint64_t> N;
  std::vector<uint64_t> rA;
  std::vector<uint64_t> rb;
  std::vector<uint64_t> rx0;

  int64_t ndim = A_cores.size();
  N.reserve(ndim);
  rA.reserve(ndim + 1);
  rb.reserve(ndim + 1);
  rx0.reserve(ndim + 1);

  // Squeeze the cores for the vector
  for (size_t i = 0; i < ndim; i++) {
    const auto& A_core = A_cores[i];
    auto& b_core = b_cores[i];

    N.push_back(A_core.size(2));
    rA.push_back(A_core.size(0));
    rb.push_back(b_core.size(0));

    if (A_core.size(1) != b_core.size(1)) {
      throw utils::runtime_error("ttnte::linalg::amen_solve",
        "The `m_modes` of `A` must equal the `m_modes` of `b`");
    }
    if (b_core.size(2) != 1) {
      throw utils::runtime_error("ttnte::linalg::amen_solve",
        "`b` must be a TT-vector with all `m_modes` equal to 1");
    }
    b_core = b_core.squeeze(2);

    if (given_x0) {
      auto& x0_core = x0_cores[i];
      rx0.push_back(x0_core.size(0));

      if (A_core.size(2) != x0_core.size(1)) {
        throw utils::runtime_error("ttnte::linalg::amen_solve",
          "The `n_modes` of `A` must equal the `m_modes` of `x0`");
      }
      if (x0_core.size(2) != 1) {
        throw utils::runtime_error("ttnte::linalg::amen_solve",
          "`x0` must be a TT-vector with all `m_modes` equal to 1");
      }
      x0_cores[i] = x0_core.squeeze(2);

    } else {
      rx0.push_back(1);
    }
  }
  rA.push_back(1);
  rb.push_back(1);
  rx0.push_back(1);

  auto x_cores = torchtt::amen_solve(A_cores, b_cores, x0_cores, N, rA, rb, rx0,
    nswp, eps, max_rank, max_full, kickrank, kick2, local_iterations, resets,
    verbose, preconditioner);

  return TTEngine(TTEngine::Tensors(x_cores.cbegin(), x_cores.cend()), true);
}

TTEngine function_interpolate(
  const std::function<torch::Tensor(const torch::Tensor&)>& func,
  const TTEngine& x, double eps, std::optional<TTEngine> start_tens, int nswp,
  int kick, int rmax, bool verbose)
{
  return python::torchtt::Acquire().function_interpolate(
    func, x, eps, start_tens, nswp, kick, rmax, verbose);
}

TTEngine function_interpolate(
  const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>& func,
  const std::vector<TTEngine>& xs, double eps,
  std::optional<TTEngine> start_tens, int nswp, int kick, int rmax,
  bool verbose)
{
  return python::torchtt::Acquire().function_interpolate(
    func, xs, eps, start_tens, nswp, kick, rmax, verbose);
}

TTEngine dmrg_cross(
  const std::function<torch::Tensor(const torch::Tensor&)>& func,
  const std::vector<int64_t>& N, double eps, int nswp,
  std::optional<TTEngine> x0, int kick, int rmax, bool verbose,
  const torch::Device& device, const torch::ScalarType& dtype)
{
  return python::torchtt::Acquire().dmrg_cross(
    func, N, eps, nswp, x0, kick, rmax, verbose, device, dtype);
}

} // namespace ttnte::linalg
