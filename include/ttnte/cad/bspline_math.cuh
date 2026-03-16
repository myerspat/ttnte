#pragma once

#include "ttnte/utils/macros.hpp"
#include <torch/headeronly/macros/Macros.h>
#include <utility>

namespace ttnte::cad::bspline {

template<typename scalar_t>
TTNTE_INLINE C10_HOST_DEVICE void basis_funcs(scalar_t u, int64_t i, int64_t p,
  const scalar_t* U, scalar_t* N, scalar_t* workspace)
{
  // Constant values
  scalar_t zero = 0.0;
  scalar_t one = 1.0;

  // Algorithm A2.2 from the NURBS book
  scalar_t* left = workspace;
  scalar_t* right = workspace + (p + 1);

  N[0] = one;

  for (int64_t j = 1; j <= p; ++j) {
    left[j] = u - U[i + 1 - j];
    right[j] = U[i + j] - u;
    scalar_t saved = zero;

    for (int64_t r = 0; r < j; ++r) {
      scalar_t divisor = right[r + 1] + left[j - r];
      scalar_t temp = (divisor > zero) ? (N[r] / divisor) : zero;

      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }

    N[j] = saved;
  }
}

template<typename scalar_t>
TTNTE_INLINE C10_HOST_DEVICE void basis_func_ders(scalar_t u, int64_t i,
  int64_t p, const scalar_t* U, scalar_t* ders_ptr, int64_t n,
  scalar_t* workspace)
{
  // Constant values
  scalar_t zero = 0.0;
  scalar_t one = 1.0;

  // Algorithm A2.3 from The NURBS Book
  // Map workspace pointers
  scalar_t* ndu = workspace;
  scalar_t* left = workspace + (p + 1) * (p + 1);
  scalar_t* right = left + (p + 1);
  scalar_t* a = right + (p + 1);

  // Index helper
  int64_t pp1 = p + 1;
  auto idx2d = [pp1](scalar_t* arr, const int64_t& i,
                 const int64_t& j) -> scalar_t& { return arr[i * pp1 + j]; };

  idx2d(ndu, 0, 0) = one;

  // Forward Pass: Compute basis functions and save knot differences
  for (int64_t j = 1; j <= p; j++) {
    left[j] = u - U[i + 1 - j];
    right[j] = U[i + j] - u;
    scalar_t saved = 0.0;

    for (int64_t r = 0; r < j; r++) {
      // Save the knot differences into the lower triangle
      idx2d(ndu, j, r) = right[r + 1] + left[j - r];
      scalar_t temp = (idx2d(ndu, j, r) > zero)
                        ? idx2d(ndu, r, j - 1) / idx2d(ndu, j, r)
                        : zero;

      idx2d(ndu, r, j) = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    idx2d(ndu, j, j) = saved;
  }

  // Store the 0-th derivative (the pure basis functions)
  for (int64_t j = 0; j <= p; j++) {
    idx2d(ders_ptr, 0, j) = idx2d(ndu, j, p);
  }

  // Compute Derivatives (Backward Substitution)
  for (int64_t r = 0; r <= p; r++) {
    int64_t s1 = 0;
    int64_t s2 = 1;
    idx2d(a, 0, 0) = one;

    for (int64_t k = 1; k <= n; k++) {
      scalar_t d = 0.0;
      int64_t rk = r - k;
      int64_t pk = p - k;

      if (r >= k) {
        idx2d(a, s2, 0) = idx2d(a, s1, 0) / idx2d(ndu, pk + 1, rk);
        d = idx2d(a, s2, 0) * idx2d(ndu, rk, pk);
      }

      int64_t j1 = (rk >= -1) ? 1 : -rk;
      int64_t j2 = (r - 1 <= pk) ? k - 1 : p - r;

      for (int64_t j = j1; j <= j2; j++) {
        idx2d(a, s2, j) =
          (idx2d(a, s1, j) - idx2d(a, s1, j - 1)) / idx2d(ndu, pk + 1, rk + j);
        d += idx2d(a, s2, j) * idx2d(ndu, rk + j, pk);
      }

      if (r <= pk) {
        idx2d(a, s2, k) = -idx2d(a, s1, k - 1) / idx2d(ndu, pk + 1, r);
        d += idx2d(a, s2, k) * idx2d(ndu, r, pk);
      }

      // Store the k-th derivative
      idx2d(ders_ptr, k, r) = d;

      // Swap rows to avoid reallocating memory
      std::swap(s1, s2);
    }
  }

  // Apply Factorial Multipliers
  // 1st deriv gets multiplied by p, 2nd by p*(p-1), 3rd by p*(p-1)*(p-2), etc.
  scalar_t r = static_cast<scalar_t>(p);
  for (int64_t k = 1; k <= n; k++) {
    for (int64_t j = 0; j <= p; j++) {
      idx2d(ders_ptr, k, j) *= r;
    }
    r *= static_cast<scalar_t>(p - k);
  }
}

} // namespace ttnte::cad::bspline
