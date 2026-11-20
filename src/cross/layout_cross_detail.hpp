#pragma once

// Private helpers of LayoutCross, shared by its source files (not
// installed API).

#include "sampling/sampling_detail.hpp"
#include "ttnte/cross/layout_cross.hpp"
#include "ttnte/cross/pivots.hpp"
#include "ttnte/linalg/banded_cholesky.hpp"
#include "ttnte/linalg/matrix_ops.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/utils/exception.hpp"
#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <c10/core/CPUAllocator.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace ttnte::cross::detail {

// The samplers' shared helpers (row threading, timing, mass density)
using sampling::detail::Clock;
using sampling::detail::DENSE_MASS_MAX;
using sampling::detail::elapsed;
using sampling::detail::row_grain;
using sampling::detail::SERIAL_ROWS;

/// Seed offset of the swept-fiber sampler: independent of the hold-out
/// draws for the same CrossConfig::seed.
inline constexpr uint64_t SAMPLE_SEED_MIX = 0x5DEECE66DULL;

inline int mode_of(tensor::Dimension dim)
{
  switch (dim) {
  case tensor::Dimension::POLAR:
    return LayoutCross::MU;
  case tensor::Dimension::AZIMUTHAL:
    return LayoutCross::GA;
  case tensor::Dimension::ENERGY:
    return LayoutCross::E;
  default:
    return LayoutCross::S;
  }
}

/// Bit mask of the cores [first, last]: a layout has at most four cores
/// (space merged into one), far below the mask's 32 bits.
inline uint32_t core_mask(size_t first, size_t last)
{
  uint32_t mask = 0;
  for (size_t k = first; k <= last; k++)
    mask |= 1u << k;
  return mask;
}

/// Merge partial components: every mode is set by exactly one part.
inline void merge(int64_t* out, const int64_t* part)
{
  for (int m = 0; m < 4; m++)
    if (part[m] >= 0)
      out[m] = part[m];
}

/// Left inverse (B^T B)^{-1} B^T of a tall B by Cholesky of its Gram matrix
/// when sigma_min / sigma_max >= min_ratio (judged on the Gram's
/// eigenvalues; that solve squares the condition number), else the
/// pseudo-inverse. `gram` is B^T B when the selection already summed it
/// (undefined: formed here).
inline torch::Tensor left_inverse(
  const torch::Tensor& B, const torch::Tensor& gram, double min_ratio)
{
  const auto G = gram.defined() ? gram : B.t().mm(B);
  const auto ev = torch::linalg_eigvalsh(G);
  const double lo = ev[0].item<double>(), hi = ev[-1].item<double>();
  if (hi > 0.0 && lo >= min_ratio * min_ratio * hi) {
    auto [L, info] = torch::linalg_cholesky_ex(G);
    if (info.item<int64_t>() == 0)
      return torch::cholesky_solve(B.t(), L);
  }
  return torch::linalg_pinv(B);
}

/// Rank-one factors u v^T of a matrix that must be an outer product (zero
/// factors for a zero matrix); undefined tensors if it is not one.
inline std::pair<torch::Tensor, torch::Tensor> outer_factors(
  const torch::Tensor& W, double tol)
{
  auto [U, s, Vh] = torch::linalg_svd(W, false);
  auto u = U.select(1, 0) * s[0];
  auto v = Vh.select(0, 0);
  if (v.sum().item<double>() < 0) {
    u = -u;
    v = -v;
  }
  const double residual = (W - u.outer(v)).norm().item<double>();
  if (residual > tol * std::max(W.norm().item<double>(), 1e-300))
    return {};
  return {u.contiguous(), v.contiguous()};
}

} // namespace ttnte::cross::detail
