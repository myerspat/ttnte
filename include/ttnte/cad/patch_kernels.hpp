#pragma once

#include "ttnte/cad/bspline_math.cuh"
#include <ATen/Dispatch.h>
#include <torch/torch.h>

namespace ttnte::cad::bspline {

inline std::tuple<torch::Tensor, torch::Tensor> launch_knot_insert_cpu(
  const torch::Tensor& Pw, // (n0+1, n1+1, ..., nk+1, ..., nd+1, dim)
  const torch::Tensor& U,  // knot vector for direction `dir`
  const torch::Tensor& X,  // knots to insert, length r+1
  int64_t p,               // degree in direction `dir`
  int64_t a, int64_t b,    // span indices
  int64_t dir)             // which parametric direction to insert along
{

  // Check they are on the same device
  TORCH_CHECK(U.device() == X.device(), "U and X must be on the same device");
  TORCH_CHECK(U.device() == Pw.device(), "U and Pw must be on the same device");

  // Check data type match
  TORCH_CHECK(U.scalar_type() == X.scalar_type() == Ubar.scalar_type(),
    "U, X, and Ubar must be the same scalar type");

  TORCH_CHECK(
    Pw.dim() >= 2, "Pw must have at least one parametric dim + component dim");
  TORCH_CHECK(dir < Pw.dim() - 1, "dir must index a parametric dimension");

  const int64_t n_param_dims = Pw.dim() - 1; // number of parametric axes
  const int64_t dim = Pw.size(-1);           // spatial components (e.g. 4)
  const int64_t n = Pw.size(dir) - 1;        // control points along dir
  const int64_t r = X.size(0) - 1;           // r+1 knots to insert

  // Build output shape: same as Pw except dir axis grows by r+1
  auto out_shape = Pw.sizes().vec();
  out_shape[dir] += (r + 1);
  auto Qw = torch::empty(out_shape, Pw.options());
  auto Ubar = torch::empty({U.size(0) + r + 1}, U.options()); // adjusted below

  auto Pw_c = Pw.contiguous();
  auto U_c = U.contiguous();
  auto X_c = X.contiguous();

  // Total number of fibers = product of all parametric dims except dir
  int64_t n_fibers = 1;
  for (int64_t ax = 0; ax < n_param_dims; ++ax) {
    if (ax != dir)
      n_fibers *= Pw.size(ax);
  }

  // Strides in the flattened fiber-index space.
  // We treat all axes except `dir` as a flat outer index, then map back
  // to per-axis indices when computing Pw/Qw offsets.
  //
  // fiber_strides[ax] = product of sizes of axes (ax+1 .. n_param_dims-1)
  //                     excluding dir, in the non-dir axis ordering.
  std::vector<int64_t> non_dir_sizes;
  non_dir_sizes.reserve(n_param_dims - 1);
  for (int64_t ax = 0; ax < n_param_dims; ++ax) {
    if (ax != dir)
      non_dir_sizes.push_back(Pw.size(ax));
  }
  std::vector<int64_t> fiber_strides(non_dir_sizes.size());
  {
    int64_t s = 1;
    for (int64_t ax = (int64_t)fiber_strides.size() - 1; ax >= 0; --ax) {
      fiber_strides[ax] = s;
      s *= non_dir_sizes[ax];
    }
  }

  // Pw strides (contiguous, including the component dim)
  // pw_stride[ax] = number of scalars to advance when ax index increments by 1
  std::vector<int64_t> pw_strides(Pw.dim());
  {
    int64_t s = 1;
    for (int64_t ax = Pw.dim() - 1; ax >= 0; --ax) {
      pw_strides[ax] = s;
      s *= Pw.size(ax);
    }
  }
  std::vector<int64_t> qw_strides(Qw.dim());
  {
    int64_t s = 1;
    for (int64_t ax = Qw.dim() - 1; ax >= 0; --ax) {
      qw_strides[ax] = s;
      s *= Qw.size(ax);
    }
  }

  AT_DISPATCH_FLOATING_TYPES(Pw.scalar_type(), "knot_insert_nd_cpu", [&] {
    const scalar_t* U_ptr = U_c.data_ptr<scalar_t>();
    const scalar_t* X_ptr = X_c.data_ptr<scalar_t>();
    const scalar_t* Pw_ptr = Pw_c.data_ptr<scalar_t>();
    scalar_t* Qw_ptr = Qw.data_ptr<scalar_t>();
    scalar_t* Ubar_ptr = Ubar.data_ptr<scalar_t>();

    // Compute Ubar once - shared across all fibers
    const int64_t m = n + p + 1;
    for (int64_t j = 0; j <= a; ++j)
      Ubar_ptr[j] = U_ptr[j];
    for (int64_t j = b + p; j <= m; ++j)
      Ubar_ptr[j + r] = U_ptr[j];

    // Parallel over fibers
    at::parallel_for(0, n_fibers, 0, [&](int64_t beg, int64_t end) {
      for (int64_t f = beg; f < end; ++f) {

        // Decode flat fiber index f -> per-axis indices (excluding dir)
        // Then compute the base pointer offset into Pw and Qw for this fiber.
        // The fiber is a 1D array of (n+1) control points along `dir`.

        int64_t pw_base = 0;
        int64_t qw_base = 0;
        {
          int64_t rem = f;
          int64_t non_dir_ax = 0;
          for (int64_t ax = 0; ax < n_param_dims; ++ax) {
            if (ax == dir)
              continue;
            int64_t idx = rem / fiber_strides[non_dir_ax];
            rem %= fiber_strides[non_dir_ax];
            pw_base += idx * pw_strides[ax];
            qw_base += idx * qw_strides[ax];
            ++non_dir_ax;
          }
        }

        // Within this fiber, consecutive control points are separated by
        // pw_strides[dir] scalars (not necessarily 1 — dir may not be last)
        const int64_t pw_step = pw_strides[dir]; // stride along dir in Pw
        const int64_t qw_step = qw_strides[dir]; // stride along dir in Qw

        patch::knot_insertion(int64_t dir, int64_t a, int64_t b, int64_t n,
          int64_t p, int64_t r, int64_t pw_step, int64_t qw_step,
          int64_t pw_base, int64_t qw_base, const scalar_t* U_ptr,
          const scalar_t* X_ptr, scalar_t* Ubar_ptr, const scalar_t* Pw_ptr,
          scalar_t* Qw_ptr);
      }
    });
  });

  return {Qw, Ubar};
}
} // namespace ttnte::cad::bspline
