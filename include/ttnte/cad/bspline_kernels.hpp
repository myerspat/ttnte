#pragma once

#include "ttnte/cad/bspline_math.cuh"
#include <ATen/Dispatch.h>
#include <torch/torch.h>

// namespace {
//
// template<typename scalar_t>
// inline void evaluate_single_point_cpu(scalar_t u, int64_t i, int64_t p,
//   const torch::TensorAccessor<scalar_t, 1>& U,
//   torch::TensorAccessor<scalar_t, 2>& N, int64_t row_idx)
// {
//   // Algorithm A2.2 from the NURBS Book
//
//   std::vector<scalar_t> left(p + 1, 0.0);
//   std::vector<scalar_t> right(p + 1, 0.0);
//   N[row_idx][0] = 1.0;
//
//   for (int64_t j = 1; j <= p; ++j) {
//     left[j] = u - U[i + 1 - j];
//     right[j] = U[i + j] - u;
//     scalar_t saved = 0.0;
//
//     for (int64_t r = 0; r < j; ++r) {
//       scalar_t divisor = right[r + 1] + left[j - r];
//       scalar_t temp = (divisor > 0.0) ? (N[row_idx][r] / divisor) : 0.0;
//
//       N[row_idx][r] = saved + right[r + 1] * temp;
//       saved = left[j - r] * temp;
//     }
//
//     N[row_idx][j] = saved;
//   }
// }
//
// template<typename scalar_t>
// inline void evaluate_single_point_cpu(scalar_t u, int64_t i, int64_t p,
//   const torch::TensorAccessor<scalar_t, 1>& U,
//   torch::TensorAccessor<scalar_t, 3>& ders, int64_t row_idx, int64_t n)
// {
//   // Algorithm A2.3 from The NURBS Book
//
//   std::vector<std::vector<scalar_t>> ndu(
//     p + 1, std::vector<scalar_t>(p + 1, 0.0));
//   std::vector<scalar_t> left(p + 1, 0.0);
//   std::vector<scalar_t> right(p + 1, 0.0);
//   std::vector<std::vector<scalar_t>> a(2, std::vector<scalar_t>(p + 1, 0.0));
//
//   ndu[0][0] = 1.0;
//
//   // Forward Pass: Compute basis functions and save knot differences
//   for (int64_t j = 1; j <= p; j++) {
//     left[j] = u - U[i + 1 - j];
//     right[j] = U[i + j] - u;
//     scalar_t saved = 0.0;
//
//     for (int64_t r = 0; r < j; r++) {
//       // Save the knot differences into the lower triangle
//       ndu[j][r] = right[r + 1] + left[j - r];
//       scalar_t temp = (ndu[j][r] > 0.0) ? ndu[r][j - 1] / ndu[j][r] : 0.0;
//
//       ndu[r][j] = saved + right[r + 1] * temp;
//       saved = left[j - r] * temp;
//     }
//     ndu[j][j] = saved;
//   }
//
//   // Store the 0-th derivative (the pure basis functions)
//   for (int64_t j = 0; j <= p; j++) {
//     ders[row_idx][0][j] = ndu[j][p];
//   }
//
//   // Compute Derivatives (Backward Substitution)
//   for (int64_t r = 0; r <= p; r++) {
//     int64_t s1 = 0;
//     int64_t s2 = 1;
//     a[0][0] = 1.0;
//
//     for (int64_t k = 1; k <= n; k++) {
//       scalar_t d = 0.0;
//       int64_t rk = r - k;
//       int64_t pk = p - k;
//
//       if (r >= k) {
//         a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
//         d = a[s2][0] * ndu[rk][pk];
//       }
//
//       int64_t j1 = (rk >= -1) ? 1 : -rk;
//       int64_t j2 = (r - 1 <= pk) ? k - 1 : p - r;
//
//       for (int64_t j = j1; j <= j2; j++) {
//         a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
//         d += a[s2][j] * ndu[rk + j][pk];
//       }
//
//       if (r <= pk) {
//         a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
//         d += a[s2][k] * ndu[r][pk];
//       }
//
//       // Store the k-th derivative
//       ders[row_idx][k][r] = d;
//
//       // Swap rows to avoid reallocating memory
//       std::swap(s1, s2);
//     }
//   }
//
//   // Apply Factorial Multipliers
//   // 1st deriv gets multiplied by p, 2nd by p*(p-1), 3rd by p*(p-1)*(p-2),
//   etc. scalar_t r = static_cast<scalar_t>(p); for (int64_t k = 1; k <= n;
//   k++) {
//     for (int64_t j = 0; j <= p; j++) {
//       ders[row_idx][k][j] *= r;
//     }
//     r *= static_cast<scalar_t>(p - k);
//   }
// }
//
// } // namespace

namespace ttnte::cad::bspline {

inline torch::Tensor launch_evaluate_cpu(const torch::Tensor& u,
  const torch::Tensor& spans, const torch::Tensor& U, int64_t p, int64_t n)
{
  // Double check the shapes of the data
  TORCH_CHECK(
    u.ndimension() == 1 && U.ndimension() == 1 && spans.ndimension() == 1,
    "u, the knotvector, and the spans should be 1-D");

  // Check they are on the same device
  TORCH_CHECK(
    u.device() == U.device(), "u and knotvector must be on the same device");
  TORCH_CHECK(
    u.device() == spans.device(), "u and span must be on the same device");

  // Check data type match
  TORCH_CHECK(u.scalar_type() == U.scalar_type(),
    "u and knotvector must be the same scalar type");

  // Make sure the derivative order and degrees are positive
  TORCH_CHECK(p >= 0 && n >= 0,
    "The polynomial degree and derivative order must be positive");

  // Enforce continuity
  auto U_c = U.contiguous();
  auto spans_c = spans.contiguous();

  // Use AT_DISPATCH_FLOATING_TYPES to handle scalar_t for any dtype support
  return AT_DISPATCH_FLOATING_TYPES(
    u.scalar_type(), "bspline_evaluate_cpu", [&] {
      // Raw data pointers
      auto u_acc = u.accessor<scalar_t, 1>();
      auto U_ptr = U_c.data_ptr<scalar_t>();
      auto spans_ptr = spans_c.data_ptr<int64_t>();

      // Run CPU multithreaded version
      if (n == 0) {
        auto N = torch::empty({u.size(0), p + 1}, u.options());

        // Result pointer
        auto N_ptr = N.data_ptr<scalar_t>();

        // Parallel run of evaluate_single_point_cpu for any scalar type
        at::parallel_for(0, u.size(0), 0, [&](int64_t start, int64_t end) {
          // Workspace for compute
          c10::SmallVector<scalar_t, 16> workspace(2 * (p + 1));

          for (int64_t k = start; k < end; k++) {
            // Specific row in result
            scalar_t* N_k = N_ptr + (k * (p + 1));

            // Execute Algorithm A2.2
            bspline::basis_funcs(
              u_acc[k], spans_ptr[k], p, U_ptr, N_k, workspace.data());
          }
        });
        return N;

      } else {
        // Clamp n
        if (n > p) {
          n = p;
        }

        // Handle derivatives
        auto ders = torch::empty({u.size(0), n + 1, p + 1}, u.options());

        // Result pointer
        auto ders_ptr = ders.data_ptr<scalar_t>();

        // Parallel run of evaluate_single_point_cpu for any scalar type
        at::parallel_for(0, u.size(0), 0, [&](int64_t start, int64_t end) {
          // Workspace for compute
          c10::SmallVector<scalar_t, 64> workspace(
            (p + 1) * (p + 1) + 4 * (p + 1));

          for (int64_t k = start; k < end; ++k) {
            // Get specific row in result
            scalar_t* ders_k = ders_ptr + (k * (n + 1) * (p + 1));

            // Execute Algorithm A2.3
            bspline::basis_func_ders(
              u_acc[k], spans_ptr[k], p, U_ptr, ders_k, n, workspace.data());
          }
        });
        return ders;
      }
    });
}

} // namespace ttnte::cad::bspline
