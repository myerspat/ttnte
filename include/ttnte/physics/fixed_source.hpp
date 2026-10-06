#pragma once

#include <functional>
#include <optional>
#include <torch/extension.h>
#include <vector>

namespace ttnte::physics {

/// @brief Specification for a fixed (static, non-eigenvalue) source term,
/// attached directly to a mesh block (volumetric) or a boundary face
/// (incident flux) before assembly. Holds only plain data -- no `linalg::`
/// types -- so it can live on `mesh::MeshBlock`/`mesh::BoundaryInfo` without
/// introducing a `mesh -> linalg` dependency.
struct FixedSource {
  /// Arbitrary function of physical-space, angle, and energy sample points
  /// (one tensor per axis, matching the assembler's own quadrature/ordinate/
  /// group sample points), evaluated via TT-cross. Used for MMS or any other
  /// source that isn't a uniform isotropic strength.
  std::optional<std::function<torch::Tensor(const std::vector<torch::Tensor>&)>>
    function = std::nullopt;
  /// Per-group isotropic source strength, in physical units (e.g.
  /// neutrons/cm^3/s for a volumetric source). No angular normalization is
  /// applied by the assembler: the angular quadrature weights are always
  /// normalized to sum to 1, so a uniform per-direction value of this
  /// strength already integrates to exactly the physical strength.
  std::optional<torch::Tensor> isotropic_strength = std::nullopt;
  /// Angle-independent, spatially varying source that is ALREADY projected
  /// onto the block's basis: b_{i,g} = integral( q_g(x) * R_i(x) ) dV, i.e.
  /// the load vector, not the NURBS coefficients of q (those differ by the
  /// mass matrix). Dense tensor of shape (n_0, ..., n_{NumDim-1}, G), with n_k
  /// the block's control-point count along parametric dimension k and G the
  /// number of energy groups. Units match `isotropic_strength`: since the
  /// angular quadrature weights sum to 1, the assembler places b on every
  /// ordinate unchanged, so the angular integral of the resulting source is
  /// exactly b. Stored dense so it is independent of the assembly format.
  std::optional<torch::Tensor> projected_isotropic_source = std::nullopt;

  /// @return True if any field is set.
  bool defined() const noexcept
  {
    return function.has_value() || isotropic_strength.has_value() ||
           projected_isotropic_source.has_value();
  }
};

} // namespace ttnte::physics
