#pragma once

#include <cstdint>

#ifndef NURBS_MAX_DERIVATIVE
#define NURBS_MAX_DERIVATIVE 5
#endif

namespace ttnte::linalg {

/// @brief A binomial table for computing cross derivatives of NURBS patches.
struct BinomialTable {
  int64_t data[NURBS_MAX_DERIVATIVE + 1][NURBS_MAX_DERIVATIVE + 1];

  constexpr BinomialTable() : data {}
  {
    for (int n = 0; n <= NURBS_MAX_DERIVATIVE; ++n) {
      data[n][0] = 1;
      for (int k = 1; k <= n; ++k) {
        data[n][k] = data[n - 1][k - 1] + data[n - 1][k];
      }
    }
  }

  inline constexpr int64_t get(int n, int k) const
  {
    return (k < 0 || k > n) ? 0 : data[n][k];
  }
};

static constexpr BinomialTable binomial;

} // namespace ttnte::linalg
