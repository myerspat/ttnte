#pragma once

#include "ttnte/sweep/patch_discretization.hpp"
#include "ttnte/sweep/sweep_config.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace ttnte::sweep {

/// @brief Maximum number of faces of a sweep unit (a 3-D hexahedral patch).
inline constexpr size_t MAX_FACES = 6;

/// @brief Where the incoming angular flux of one face comes from, for the
/// directions of one sweep task.
struct FaceInflow {
  enum class Kind : uint8_t {
    /// No incoming flux (vacuum, degenerate, or no flow for these
    /// directions).
    NONE,
    /// Upwind trace of a neighboring unit.
    NEIGHBOR,
    /// Specular reflection of this unit's own outgoing trace.
    REFLECTIVE,
  };
  Kind kind = Kind::NONE;
  /// (N_Omega, G, source_face_dofs) contiguous trace buffer to read.
  const double* trace = nullptr;
  /// Number of dofs on the source face (the neighbor's face for NEIGHBOR).
  int64_t source_face_dofs = 0;
  /// NEIGHBOR: (Nf) source-face index of each of this face's dofs.
  const int64_t* trace_map = nullptr;
  /// REFLECTIVE: (N_Omega) reflected direction of each direction.
  const int64_t* reflected = nullptr;
  /// REFLECTIVE: specular albedo.
  double albedo = 1.0;
};

/// @brief Raw views of everything one unit's sweep reads and writes.
struct UnitSweep {
  const PatchDiscretization* disc = nullptr;
  /// (G) total cross section.
  const double* sigma_t = nullptr;
  /// (num_moments, G, N) scattering + external source moments.
  const double* source_moments = nullptr;
  /// Incoming flux per face (indexed 2 * dim + is_upper).
  std::array<FaceInflow, MAX_FACES> inflow {};
  /// Per face (N_Omega, G, Nf) outgoing trace buffers, or nullptr.
  std::array<double*, MAX_FACES> trace_out {};
  /// (N_Omega, G, N) angular flux, or nullptr if not stored.
  double* psi = nullptr;
  /// (num_moments, num_block_groups, N) moments ACCUMULATED by the sweep
  /// (the caller zeroes and reduces it).
  double* moments = nullptr;
  /// Optional factorization cache: (N_Omega, G, matrix size) factors and
  /// (N_Omega, G, N) pivots, matrix size being N * ldab (banded) or N * N
  /// (dense). nullptr disables caching.
  double* factors = nullptr;
  int64_t* factor_pivots = nullptr;
  /// Whether `factors` already holds this unit's factors (otherwise they
  /// are computed and stored).
  bool factors_ready = false;
};

/// @brief The directions, groups and angular data of one sweep task.
struct SweepBlock {
  /// Directions and groups swept by the task.
  const int64_t* directions = nullptr;
  int64_t num_block_directions = 0;
  const int64_t* groups = nullptr;
  int64_t num_block_groups = 0;
  /// Totals (buffer strides).
  int64_t num_directions = 0;
  int64_t num_groups = 0;
  int64_t num_moments = 0;
  /// (N_Omega, ndim) streaming directions.
  const double* omega = nullptr;
  /// (N_Omega, num_moments) harmonics, and the same times the weights.
  const double* harmonics = nullptr;
  const double* weighted_harmonics = nullptr;
  /// Local solver.
  LocalSolverType solver = LocalSolverType::BANDED_LU;
};

/// @brief Scratch memory reused across units and tasks (one per worker).
struct SweepWorkspace {
  std::vector<double> values;
  std::vector<double> matrix;
  std::vector<double> rhs;
  std::vector<double> inflow;
  std::vector<int64_t> pivots;
};

/// @brief Sweep one unit for every (direction, group) of a task: assemble
/// L_{d,g} = sigma_{t,g} M - Omega_d . G + sum_f B_f^+(d) from the unit's
/// pair lists, add the upwind face inflow to the scattering + external
/// source, factor and solve with the configured local solver, and write
/// the outgoing face traces, the angular flux (if requested) and the moment
/// contributions.
///
/// Plain loops over raw arrays: the per-unit work of an FDG element is a
/// few microseconds of arithmetic, far below the fixed cost of a tensor
/// framework call. Thread-safe as long as concurrent calls write disjoint
/// directions/groups of the shared buffers and use their own workspace and
/// moment buffers.
/// @throws std::runtime_error If a local system is singular.
void sweep_unit(
  const UnitSweep& unit, const SweepBlock& block, SweepWorkspace& workspace);

/// @brief Scattering + external source moments of one unit, in raw loops:
///   out_{m,g} = M sum_g' sigma_{l(m), g <- g'} phi_{m,g'}  (+ q_ext for m = 0)
/// (the weak-form source (sigma_s phi, R_v): phi shares the unit's basis, so
/// the mass matrix applies it).
/// @param view The unit's discretization.
/// @param scatter (L, G, G) scattering matrices (rows destination), or
/// nullptr for none.
/// @param num_orders L.
/// @param moment_order (num_moments) Legendre order of each moment.
/// @param num_moments Number of moments.
/// @param num_groups G.
/// @param moments (num_moments, G, N) current flux moments.
/// @param external (G, N) external source load vector, or nullptr.
/// @param out (num_moments, G, N) output source moments.
/// @param scratch Reused scratch (resized as needed).
void scattering_source(const KernelView& view, const double* scatter,
  int64_t num_orders, const int64_t* moment_order, int64_t num_moments,
  int64_t num_groups, const double* moments, const double* external,
  double* out, std::vector<double>& scratch);

} // namespace ttnte::sweep
