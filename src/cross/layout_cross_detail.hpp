#pragma once

// Private helpers of LayoutCross, shared by its source files (not
// installed API).

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

using Clock = std::chrono::steady_clock;

// Performance settings: they change speed or memory, never the results
// beyond round-off, and are machine dependent, so they are build options
// (CMake cache variables of the same name, see CMakeLists.txt).
#ifndef TTNTE_DENSE_MASS_MAX
#define TTNTE_DENSE_MASS_MAX 3000
#endif
#ifndef TTNTE_SERIAL_ROWS
#define TTNTE_SERIAL_ROWS 64
#endif
#ifndef TTNTE_ROW_CHUNKS_PER_THREAD
#define TTNTE_ROW_CHUNKS_PER_THREAD 4
#endif
#ifndef TTNTE_ROW_GRAIN_MIN
#define TTNTE_ROW_GRAIN_MIN 8
#endif

/// Largest patch whose mass Cholesky factor is also kept dense (n^2 doubles,
/// for BLAS-3 weighting of fiber batches).
inline constexpr int64_t DENSE_MASS_MAX = TTNTE_DENSE_MASS_MAX;
/// Fewer fiber rows than this are weighted serially (the sweep's callbacks
/// run on its worker threads).
inline constexpr int64_t SERIAL_ROWS = TTNTE_SERIAL_ROWS;

/// Seed offset of the swept-fiber sampler: independent of the hold-out
/// draws for the same CrossConfig::seed.
inline constexpr uint64_t SAMPLE_SEED_MIX = 0x5DEECE66DULL;

/// Grain of a parallel loop over n fiber rows: TTNTE_ROW_CHUNKS_PER_THREAD
/// chunks per thread, at least TTNTE_ROW_GRAIN_MIN rows each, every chunk
/// serial inside (small per-row operations do not thread well).
inline int64_t row_grain(int64_t n)
{
  return std::max<int64_t>(TTNTE_ROW_GRAIN_MIN,
    n / std::max<int64_t>(
          1, TTNTE_ROW_CHUNKS_PER_THREAD * at::get_num_threads()));
}

inline double elapsed(Clock::time_point t0)
{
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

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
