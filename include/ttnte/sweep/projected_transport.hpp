#pragma once

#include "ttnte/sweep/sweep_driver.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <utility>
#include <vector>

namespace ttnte::sweep {

/// @brief Galerkin projection of one group set's transport operator onto a
/// growing basis of group-set fluxes, per direction (Guo & Peng, 2026,
/// eq. 3.4, multigroup by angular augmentation over the set's groups).
///
/// A basis vector x_j is one flux over the set's groups and every local
/// unit, in the fiber layout (num_groups, fiber_length), unweighted. For
/// direction d the discrete transport problem is A_d psi_d = b_d, with
///   A_d x = L_d x - (inflow of x across the faces d sweeps upwind first),
///   b_d   = sum_m Y_m(d) s_m + (inflow of the lagged and reflected traces),
/// exactly as SweepSolver::transport_residual() assembles r = b_d - A_d x.
/// The reduced system is X^T A_d X c = X^T b_d (rows of X the basis), with
///   X^T A_d X = C - sum_i Omega_i S_i
///             + sum_n (Omega . n)_+ F_n - sum_n (Omega . n)_- K_{a(d),n}
///             + D_d,
/// all bordered as the basis grows (eq. 3.10) rather than recomputed:
/// - C = sum_g sum_u sigma_{t,u,g} X^T M_u X (collision) and S_i =
///   sum_g X^T G_i X (streaming), independent of direction;
/// - faces whose unit normal is constant (flat): F_n sums the outflow
///   integrals sum_q |n_q| w_q (R X)_q^T (R X)_q of every face with unit
///   normal n, and K_{a,n} the inflow integrals sum_q |n_q| w_q (R X)_q^T
///   (R^nb X^nb)_q of the faces with normal n that angle set a sweeps
///   upwind first (DEPENDENCY), so a direction costs one r x r term per
///   distinct normal;
/// - curved faces: D_d, their outflow and dependency inflow integrals at
///   direction d's clipped normals, cached per direction and bordered
///   lazily when the direction is next assembled;
/// - X^T s_m per moment, and the lagged and reflected inflow projected per
///   direction through the sweep kernel's own inflow (buffered_inflow()).
/// The sources and lagged traces are read at extend() and reduced_system(),
/// so a basis belongs to one transport solve (reset() between solves).
class ProjectedTransport {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<ProjectedTransport>;

  struct Options {
    /// Take the flat-face path (one matrix per distinct normal) where a
    /// face's unit normal is constant; false: every face by quadrature.
    bool flat_faces = true;
    /// Treat the LAGGED internal edges (cycle breaks) as DEPENDENCY: their
    /// inflow enters A_d from the basis instead of b_d from the buffers.
    /// The sweep cannot do this (it needs an acyclic order) but a reduced
    /// solve can; off by default so that A_d matches the oracle (testing
    /// aid, and a possible labelled ablation).
    bool implicit_lagged = false;
  };

protected:
  // =================================================================
  // Protected types
  /// @brief A matrix whose rows and columns grow (capacity doubles, the
  /// contents kept).
  struct Growing {
    torch::Tensor storage;
    int64_t rows = 0;
    int64_t cols = 0;
    void resize(int64_t num_rows, int64_t num_cols);
    torch::Tensor view() const
    {
      return storage.narrow(0, 0, rows).narrow(1, 0, cols);
    }
  };

  /// @brief Projections of one face of one unit.
  struct FaceProjection {
    size_t unit = 0;
    size_t face = 0;
    /// INTERNAL: the upwind neighbor (local unit) and its face.
    bool internal = false;
    size_t neighbor = 0;
    size_t neighbor_face = 0;
    /// Per angle set: whether the set sweeps the neighbor first.
    std::vector<uint8_t> dependency;
    /// Index into normals_ of a flat face's unit normal, -1 if curved.
    int64_t normal = -1;
    /// Flat: (num_groups Qf) |n_q| w_q, repeated per group.
    torch::Tensor measure;
    /// (num_groups Qf, r) the basis at the face's quadrature points, (R
    /// X)_{g,q}, row g * Qf + q; INTERNAL: the neighbor's, (R^nb X^nb)_{g,q}.
    Growing own;
    Growing upwind;
  };

  // =================================================================
  // Protected data
  const SweepSolver& solver_;
  std::vector<int64_t> groups_;
  Options options_;
  /// Groups of the projected set, groups_.size().
  int64_t num_groups_ = 0;
  int64_t num_space_ = 0;
  int64_t ndim_ = 0;
  int64_t num_moments_ = 0;
  int64_t num_directions_ = 0;
  int64_t num_sets_ = 0;
  int64_t rank_ = 0;

  /// (r, num_groups, num_space) basis (rows of storage).
  Growing basis_;
  /// (r, r) collision and per dimension streaming projections.
  Growing collision_;
  std::vector<Growing> streaming_;
  /// (num_moments, r) X^T s_m.
  Growing source_;
  std::vector<FaceProjection> faces_;
  /// Distinct unit normals of the flat faces, (ndim) each, their outflow
  /// projections F_n, and per angle set their dependency inflow K_{a,n}.
  std::vector<torch::Tensor> normals_;
  std::vector<Growing> outflow_;
  std::vector<std::vector<Growing>> inflow_;
  /// Curved faces (indices into faces_), and per direction their cached
  /// projection D_d (bordered up to its own size).
  std::vector<size_t> curved_;
  std::vector<Growing> curved_cache_;
  /// Per angle set, the (unit, face) pairs whose inflow comes from a buffer
  /// (lagged edges, reflective faces).
  std::vector<std::vector<std::pair<size_t, size_t>>> buffered_;

  // =================================================================
  // Protected constructors
  /// @throws std::runtime_error On more than one rank, or if `group_set`
  /// is out of range.
  ProjectedTransport(
    const SweepSolver& solver, size_t group_set, Options options);

  [[nodiscard]] static std::string ctx(const std::string& f)
  {
    return "ttnte::sweep::ProjectedTransport::" + f;
  }

  /// @brief Border D_d up to the current rank.
  void update_curved(int64_t d);

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new ProjectedTransport(std::forward<Args>(args)...));
  }

  /// @brief Drop the basis (a new transport solve: new sources and traces).
  void reset();

  /// @brief Append basis vectors and border every projection.
  /// @param X (p, num_groups, fiber_length) new basis vectors (unweighted, the
  /// set's groups in order).
  /// @throws std::runtime_error On a wrong shape.
  void extend(const torch::Tensor& X);

  /// @brief The reduced system of direction d, X^T A_d X and X^T b_d.
  /// Directions may be assembled concurrently (each borders only its own
  /// cache).
  /// @return (r, r) matrix and (r) right-hand side.
  /// @throws std::runtime_error If d is out of range.
  std::pair<torch::Tensor, torch::Tensor> reduced_system(int64_t d);

  /// @brief The part of X^T b_d that is data rather than source: the
  /// inflow of direction d read from the buffers (lagged internal edges of
  /// d's angle set, reflective faces), projected.
  /// @return (r) X^T (lagged and reflected inflow of direction d).
  /// @throws std::runtime_error If d is out of range.
  torch::Tensor buffered_rhs(int64_t d) const;

  // =================================================================
  // Public getters
  int64_t rank() const noexcept { return rank_; }
  /// @return (r, num_groups, fiber_length) the basis.
  torch::Tensor basis() const;
  const std::vector<int64_t>& groups() const noexcept { return groups_; }
  /// @return Distinct unit normals of the flat faces.
  int64_t num_normals() const noexcept
  {
    return static_cast<int64_t>(normals_.size());
  }
  /// @return Faces projected by quadrature per direction.
  int64_t num_curved_faces() const noexcept
  {
    return static_cast<int64_t>(curved_.size());
  }
};

} // namespace ttnte::sweep
