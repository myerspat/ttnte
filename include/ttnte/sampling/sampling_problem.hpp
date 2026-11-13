#pragma once

#include "ttnte/sweep/fiber_block.hpp"
#include <cstdint>
#include <functional>
#include <torch/extension.h>
#include <vector>

namespace ttnte::sampling {

/// @brief The discrete transport problem as seen by a sampler: index sizes,
/// angular data, the spatial blocks (patches) of a fiber and the oracle that
/// sweeps fibers.
struct SamplingProblem {
  /// Sweeps a request of fibers into a (num_fibers, num_space) buffer, block
  /// by block, direction-major then group (see SweepDriver::sweep_fibers()),
  /// calling `done` exactly once for every output row (in ranges, possibly
  /// concurrently, as fibers finish).
  using SweepFn =
    std::function<void(const std::vector<sweep::FiberBlock>& blocks,
      double* out, const sweep::FiberDone& done)>;

  int64_t num_dim = 0;
  int64_t num_polar = 0;
  /// 1 in 1-D. Direction d = polar * num_azimuthal + azimuthal.
  int64_t num_azimuthal = 1;
  int64_t num_groups = 0;
  /// Fiber length (dofs of every patch, concatenated).
  int64_t num_space = 0;
  /// (N_Omega) quadrature weights.
  torch::Tensor weights;
  /// (N_Omega, num_moments) harmonics; moments are sum_d w_d Y_m(d) psi_d.
  torch::Tensor harmonics;
  /// Directions swept over all groups to start a sampler (one per
  /// quadrant/octant): the cross always, the Galerkin sampler with
  /// GalerkinStart::INITIAL_DIRECTIONS; at least one, all in range.
  std::vector<int64_t> initial_directions;
  /// Offset of each patch in a fiber, and its control-point grid shape.
  std::vector<int64_t> patch_offsets;
  std::vector<std::vector<int64_t>> patch_shapes;
  /// Per patch, the lower band (n_P, kd + 1) of its mass matrix (L2
  /// weighting in space); empty for no spatial weighting.
  std::vector<torch::Tensor> mass_bands;
  /// (num_groups) lethargy widths ln(E_{g-1} / E_g), all positive, for the
  /// energy weighting 1 / sqrt(du_g); empty for uniform weighting.
  std::vector<double> lethargy_widths;
  SweepFn sweep;
};

} // namespace ttnte::sampling
