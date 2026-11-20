#pragma once

// Private helpers of the samplers (FiberStore and every sampler built on
// it), shared by their source files (not installed API).

#include <ATen/Parallel.h>
#include <algorithm>
#include <chrono>
#include <cstdint>

namespace ttnte::sampling::detail {

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

/// Grain of a parallel loop over n fiber rows: TTNTE_ROW_CHUNKS_PER_THREAD
/// chunks per thread, at least TTNTE_ROW_GRAIN_MIN rows each, every chunk
/// serial inside (small per-row operations do not thread well).
inline int64_t row_grain(int64_t n)
{
  return std::max<int64_t>(TTNTE_ROW_GRAIN_MIN,
    n / std::max<int64_t>(
          1, TTNTE_ROW_CHUNKS_PER_THREAD * at::get_num_threads()));
}

/// Grain of a parallel loop over n fiber rows: one chunk (serial) below
/// SERIAL_ROWS, else row_grain(n).
inline int64_t loop_grain(int64_t n)
{
  return n < SERIAL_ROWS ? std::max<int64_t>(n, 1) : row_grain(n);
}

inline double elapsed(Clock::time_point t0)
{
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

} // namespace ttnte::sampling::detail
