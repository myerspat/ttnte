#pragma once

#include "ttnte/utils/macros.hpp"
#include <torch/headeronly/macros/Macros.h>
#include <utility>

namespace ttnte::cad::bspline {

// Knot Insertion Algorithim
template<typename scalar_t>
TTNTE_INLINE C10_HOST_DEVICE void knot_insertion(int64_t dir, int64_t a,
  int64_t b, int64_t n, int64_t p, int64_t r, int64_t pw_step, int64_t qw_step,
  int64_t pw_base, int64_t qw_base, const scalar_t* U_ptr,
  const scalar_t* X_ptr, scalar_t* Ubar_ptr, const scalar_t* Pw_ptr,
  scalar_t* Qw_ptr)
{
  // Algorithm A5.5 from the NURBS book
  // int64_t a = find_spans(param_idx, new_coords[0]).item<int64_t>();
  // int64_t b = find_spans(param_idx, new_coords[-1]).item<int64_t>() + 1;

  int64_t m = n + p + 1;

  // Lambda to get pointer to Pw[..., j, ..., :] for this fiber
  auto pw_cp = [&](int64_t j) -> const scalar_t* {
    return Pw_ptr + pw_base + j * pw_step;
  };
  auto qw_cp = [&](int64_t j) -> scalar_t* {
    return Qw_ptr + qw_base + j * qw_step;
  };

  // ---- A5.5 on this fiber ----------------------------------------

  // Copy unaffected left band
  for (int64_t j = 0; j <= a - p; ++j) {
    const scalar_t* src = pw_cp(j);
    scalar_t* dst = qw_cp(j);
    for (int64_t c = 0; c < dim; ++c)
      dst[c] = src[c];
  }

  // Copy unaffected right band
  for (int64_t j = b - 1; j <= n; ++j) {
    const scalar_t* src = pw_cp(j);
    scalar_t* dst = qw_cp(j + r + 1);
    for (int64_t c = 0; c < dim; ++c)
      dst[c] = src[c];
  }

  int64_t i = b + p - 1;
  int64_t k = b + p + r;

  for (int64_t j = r; j >= 0; --j) {
    while (X_ptr[j] <= U_ptr[i] && i > a) {
      Ubar_ptr[k] = U_ptr[i];
      const scalar_t* src = pw_cp(i - p - 1);
      scalar_t* dst = qw_cp(k - p - 1);
      for (int64_t c = 0; c < dim; ++c)
        dst[c] = src[c];
      k--;
      i--;
    }

    // Shift: Qw[k-p-1] = Qw[k-p]
    {
      const scalar_t* src = qw_cp(k - p);
      scalar_t* dst = qw_cp(k - p - 1);
      for (int64_t c = 0; c < dim; ++c)
        dst[c] = src[c];
    }

    for (int64_t l = 1; l <= p; ++l) {
      int64_t ind = k - p + l;
      scalar_t alpha = Ubar_ptr[k + l] - X_ptr[j];

      scalar_t* P0 = qw_cp(ind - 1);
      const scalar_t* P1 = qw_cp(ind);

      if (std::abs(alpha) == static_cast<scalar_t>(0)) {
        for (int64_t c = 0; c < dim; ++c)
          P0[c] = P1[c];
      } else {
        alpha /= (Ubar_ptr[k + l] - U_ptr[i - p + l]);
        for (int64_t c = 0; c < dim; ++c)
          P0[c] = alpha * P0[c] + (static_cast<scalar_t>(1) - alpha) * P1[c];
      }
    }

    Ubar_ptr[k] = X_ptr[j];
    k--;
  }
}

// Bezier Decomposition Algorithim
template<typename scalar_t>
TTNTE_INLINE C10_HOST_DEVICE void bezier_decomposition(int64_t n, int64_t p,
  const torch::TensorAccessor<scalar_t, 1> U, const torch::Tensor Pw,
  int64_t nb, int64_t dir, torch::Tensor Qw),
  std::vector<scalar_t> alphas
{
  // the first dimension of Pw tracks which index of a bezier curve
  int64_t m = n + p + 1;
  int64_t a = p;
  int64_t b = p + 1;
  nb = 0;

  for (int64_t i = 0; i <= p; ++i) {
    Qw.select(0, nb).select(dir, i) = Pw.select(dir, i);
  }

  while (b < m) {
    int64_t mult = b - i + 1;
    if (mult < p) {

      scalar_t numer = U[b] - U[a];

      for (int64_t j = p; j > mult; --j) {
        alphas[j - mult - 1] =
          numer / (U[a + j] - U[a]); // check typing on this
      }

      int64_t r = p - mult;

      for (int64_t j = 1; j <= p - mult; ++j) {

        int64_t save = r - j;
        int64_t s = mult + j;

        for (int64_t k = p; k >= s; --k) {

          int64_t alpha = alphas[k - s];
          Qw.select(0, nb).select(dir, k) =
            alpha * Qw.select(0, nb).select(dir, k) +
            (1.0 - alpha) * Qw.select(0, nb).select(dir, k - 1);
        }

        if (b < m) {
          Qw.select(0, nb + 1).select(dir, save) =
            Qw.select(0, nb).select(dir, p);
        }
      }

      nb = nb + 1;

      if (b < m) {

        for (int64_t i = p - mult; i <= p; ++i) {
          Qw.select(0, nb).select(dir, i) = Pw.select(dir, b - p + i);
        }

        a = b;
        b = b + 1;
      }
    }
  }
}

// Knot Removal
template<typename scalar_t>
TTNTE_INLINE C10_HOST_DEVICE void knot_removal(int64_t n, int64_t p,
  const torch::TensorAccessor<scalar_t, 1> U, const torch::Tensor Pw,
  int64_t nb, torch::Tensor Qw),
  std::vector<scalar_t> alphas
{
  int64_t m = n + p + 1;
}

// I need to check type ambivalence, and also test
} // namespace ttnte::cad::bspline
