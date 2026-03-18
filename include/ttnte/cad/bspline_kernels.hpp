#pragma once

#include "ttnte/cad/bspline_math.cuh"
#include <torch/extension.h>

namespace ttnte::cad::bspline {

inline torch::Tensor launch_evaluate_cpu(const at::Tensor& u,
  const torch::Tensor& spans, const at::Tensor& U, int64_t p, int64_t n)
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
