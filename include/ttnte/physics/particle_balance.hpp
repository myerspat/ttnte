#pragma once

#include "ttnte/physics/boundary_types.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <torch/extension.h>
#include <vector>

namespace ttnte::physics {

/// @brief Particle-balance diagnostics for a single boundary face, computed
/// entirely from the owning patch's own solution vector -- no data is shared
/// across patches to build this. `outgoing`/`incoming` are per-group partial
/// currents through this face (length `num_groups`).
struct FaceBalance {
  /// The dimension of the face.
  size_t dim;
  /// Whether the face is at the upper or lower end of `dim`.
  bool is_upper;
  /// The boundary condition type of this face.
  BoundaryType type;
  /// Set only for `BoundaryType::INTERNAL` -- the global ID of the
  /// neighboring patch sharing this face.
  std::optional<int64_t> neighbor_gid;
  /// Set only for `BoundaryType::INTERNAL` -- the physical dimension of the
  /// neighboring patch's own face on this shared interface (see
  /// `mesh::NeighborInfo::dim`). Needed to find the matching `FaceBalance`
  /// on the neighbor's side, since a patch pair's shared interface is not
  /// guaranteed to sit at the same `dim`/opposite `is_upper` on both sides
  /// (e.g. rotated/non-axis-aligned NURBS patches).
  std::optional<size_t> neighbor_dim;
  /// Set only for `BoundaryType::INTERNAL` -- the orientation of the
  /// neighboring patch's own face on this shared interface (see
  /// `mesh::NeighborInfo::is_upper`), paired with `neighbor_dim`.
  std::optional<bool> neighbor_is_upper;
  /// Outgoing (this patch's own) partial current through this face.
  torch::Tensor outgoing;
  /// Incoming partial current through this face. Set for `REFLECTIVE`
  /// (albedo * the mirrored outgoing current, i.e. the assembled,
  /// albedo-scaled reflective inflow operator applied to psi and integrated
  /// -- equals albedo * `outgoing` for a mirror-symmetric quadrature) and
  /// `INCIDENT` (from the prescribed source), both computed purely locally.
  /// Left unset for `VACUUM` (identically 0, not computed). For `INTERNAL`,
  /// computed purely locally as unset here -- it is instead resolved to the
  /// neighbor's own `outgoing` (via
  /// `neighbor_gid`/`neighbor_dim`/`neighbor_is_upper`) after every patch has
  /// been gathered, by
  /// `TransportSolution::patch_balance_table()`/`global_balance()`'s
  /// `resolve_internal_faces()`, which also folds the result into this
  /// patch's `leakage`. Left unset here too if the neighbor patch isn't
  /// present in the gathered table (shouldn't happen in a normal solve).
  std::optional<torch::Tensor> incoming;
};

/// @brief Particle-balance diagnostics for a single patch. Every tensor is
/// length `num_groups`. Per-group balance:
/// `leakage + absorption + scatter_out == scatter_in + fission_source +
/// fixed_source` (up to solve tolerance) -- true as soon as this comes out of
/// `TransportSolution::compute_patch_balances()` for every quantity except
/// `leakage`'s `INTERNAL` contributions (deferred -- see `FaceBalance`), and
/// fully true (including `INTERNAL` faces) once resolved by
/// `patch_balance_table()`/`global_balance()`'s `resolve_internal_faces()`.
struct PatchBalance {
  /// The global ID of the patch this balance was computed for.
  int64_t gid;
  /// Total prescribed (volumetric + incident-boundary) fixed source.
  torch::Tensor fixed_source;
  /// Total fission source (`chi ⊗ nu_fission` reaction rate).
  torch::Tensor fission_source;
  /// Total absorption reaction rate.
  torch::Tensor absorption;
  /// Total scattering-in reaction rate (isotropic/l=0 moment only -- see
  /// `DGFirstOrderTransportBackend::assemble_balance_functional()`).
  torch::Tensor scatter_in;
  /// Total scattering-out (removal) reaction rate.
  torch::Tensor scatter_out;
  /// Sum over every face of (`outgoing` - `incoming`, or bare `outgoing` for
  /// `VACUUM` and for any `INTERNAL` face not yet resolved) -- this patch's
  /// own view of its net leakage. `INTERNAL` faces contribute nothing here
  /// until `resolve_internal_faces()` folds them in (see `FaceBalance`).
  torch::Tensor leakage;
  /// Per-face breakdown, one entry per boundary face.
  std::vector<FaceBalance> faces;
};

/// @brief Every patch's own particle-balance diagnostics, gathered across
/// every rank and sorted by GID -- identical on every rank. Returned by
/// `TransportSolution::patch_balance_table()`. A thin wrapper around
/// `std::vector<PatchBalance>` (rather than binding the vector directly) so
/// it has its own Python type to attach a `to_dataframe()` method to (see
/// `ttnte/physics/_particle_balance.py`), the same way plotting is attached
/// to `ttnte.cad.Patch`.
struct PatchBalanceTable {
  /// Every patch's PatchBalance, sorted by `gid`.
  std::vector<PatchBalance> patches;
};

/// @brief Particle-balance diagnostics summed over the whole problem.
struct GlobalBalance {
  torch::Tensor fixed_source;
  torch::Tensor fission_source;
  torch::Tensor absorption;
  torch::Tensor scatter_in;
  torch::Tensor scatter_out;
  /// Sum of (`outgoing` - `incoming`) over every `VACUUM`/`REFLECTIVE`/
  /// `INCIDENT` face across every patch -- the true, whole-system leakage.
  /// Computed as a straight sum of every patch's own (resolved) `leakage`;
  /// each `INTERNAL` interface's two sides cancel out exactly (both are
  /// resolved from the same pair of `outgoing` tensors, just negated), so no
  /// separate exclusion is needed here.
  torch::Tensor leakage;
  /// Sum, over every `INTERNAL` interface (each counted once), of
  /// `|this_side.outgoing - adjacent_side.outgoing|` -- the two patches'
  /// independently, locally computed outgoing currents through their shared
  /// face. Should be ~0 for a lossless, fully converged distributed solve;
  /// a nonzero value directly indicates the DD exchange is losing (or
  /// gaining) particles at patch interfaces.
  torch::Tensor dd_residual;
};

} // namespace ttnte::physics
