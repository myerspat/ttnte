#pragma once

#include "ttnte/galerkin/reduced_operator.hpp"
#include "ttnte/sweep/sweep_driver.hpp"
#include <cstdint>
#include <functional>
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
/// - curved faces: their outflow and dependency inflow integrals at
///   direction d's clipped normals. The clipped weights (Omega . n_q)_+ and
///   (-Omega . n_q)_+ are linear in Omega among the directions that give
///   every quadrature point of the face the same sign of Omega . n_q, so
///   each face keeps one bordered matrix per such sign class and dimension
///   (sign_split, a direction then costs ndim r x r terms per face); a face
///   with more than max_face_classes classes, or every face when the class
///   matrices outnumber the directions (max_class_ratio), keeps the
///   direction's own
///   projection D_d, cached per direction and bordered lazily when the
///   direction is next assembled;
/// - X^T s_m per moment, and the lagged and reflected inflow projected per
///   direction through the sweep kernel's own inflow (buffered_inflow()).
/// The sources and lagged traces are read at extend() and reduced_system(),
/// so a basis belongs to one transport solve (reset() between solves).
///
/// With Options::scattering it also projects the within-set scattering
/// S_l (S_l x)_g = sum_{g' in set} sigma_{s,l,u}(g <- g') M_u x_{g'} per
/// Legendre order, K_l = X^T S_l X, and solves Han et al. (2026)'s angular
/// Galerkin problem (angular_galerkin(): eq. 3.4, every direction coupled
/// through the reduced scattering moments).
///
/// solve() and residuals() complete the galerkin::ReducedOperator: per
/// direction an LU of the r x r system (LAPACK, in parallel over
/// directions), and the residuals of X c_d for a batch of directions,
/// matrix-free with the same volume pairs and face traces (equal to the
/// sweep kernel's transport_residual()).
class ProjectedTransport : public galerkin::ReducedOperator {
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
    /// Project each curved face once per sign class of Omega . n_q (exact:
    /// within a class the clipped weights are linear in Omega) instead of
    /// once per direction. False: every curved face per direction.
    bool sign_split = true;
    /// Curved faces with more sign classes than this keep the per-direction
    /// projection (classes cost memory ndim r^2 each; in 2-D a face has
    /// about one per quadrature point, in 3-D possibly one per direction).
    int64_t max_face_classes = 64;
    /// Split only while the class matrices (classes x ndim per curved face,
    /// twice on internal faces) number at most this times the directions;
    /// otherwise every curved face keeps the per-direction projection (one
    /// cached matrix per direction). Past that, the class matrices cost more
    /// memory and bordering than the per-direction caches, and a direction's
    /// assembly sums more terms than one cached matrix (vnv reflected
    /// cylinder: 28 faces x 34 classes against 1024 directions, 10x the
    /// bordering).
    double max_class_ratio = 1.0;
    /// Pair sampling (GalerkinConfig::sample = PAIR): a spatial basis, one
    /// group per vector, and one reduced system per (direction, group),
    /// unit u = d num_groups + g. The streaming and face projections do not
    /// depend on the group and are shared; the collision (sigma_{t,g}), the
    /// sources and the buffered inflow are per group. False: one system per
    /// direction over the set's groups (Guo & Peng's multigroup).
    bool spatial_basis = false;
    /// Also project the within-set scattering, K_l = X^T S_l X per Legendre
    /// order (bordered with the volume terms in extend()), for
    /// project_scattering() and angular_galerkin() (Han et al. 2026's
    /// angular update). Off: extend() costs as before. Direction sampling
    /// only.
    bool scattering = false;
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
    /// Flat: (basis_groups Qf) |n_q| w_q, repeated per basis group.
    torch::Tensor measure;
    /// Its rows [row0, row0 + rows) of the global traces (rows =
    /// basis_groups Qf): the basis at the face's quadrature points, (R
    /// X)_{g,q}, row
    /// row0 + g * Qf + q (traces_own_); INTERNAL: the neighbor's, (R^nb
    /// X^nb)_{g,q} (traces_upwind_, zero rows otherwise).
    int64_t row0 = 0;
    int64_t rows = 0;
  };

  /// @brief A curved face projected per sign class (Options::sign_split):
  /// the directions that give each quadrature point q the same sign of
  /// Omega . n_q. Within a class, (Omega . n_q)_+ w_q = sum_i Omega_i
  /// n_{q,i} w_q on the positive points and (-Omega . n_q)_+ w_q = -sum_i
  /// Omega_i n_{q,i} w_q on the negative ones, so the face's terms of D_d
  /// are sum_i Omega_i (O_{c,i} + I_{c,i}), with
  ///   O_{c,i} = (T X)^T diag(n_i w [q positive in c]) (T X)      outflow
  ///   I_{c,i} = (T X)^T diag(n_i w [q negative in c]) (T^nb X)   inflow
  /// (n_q unnormalized, the kernel's upwind weights; I only for INTERNAL
  /// faces, and added only for the angle sets that sweep the neighbor
  /// first).
  struct FaceClasses {
    /// Index into faces_.
    size_t face = 0;
    int64_t num_classes = 0;
    /// Per direction its class.
    std::vector<int64_t> class_of;
    /// (classes, Qf) 1 on the positive (negative) points of each class.
    torch::Tensor positive;
    torch::Tensor negative;
    /// (ndim, Qf) n_{q,i} w_q.
    torch::Tensor moments;
    /// (classes, ndim, capacity, capacity) O and, for INTERNAL faces, I;
    /// the leading (rank, rank) block in use.
    torch::Tensor outflow;
    torch::Tensor inflow;
  };

  // =================================================================
  // Protected data
  const SweepSolver& solver_;
  std::vector<int64_t> groups_;
  Options options_;
  /// Groups of the projected set, groups_.size(); groups of a basis vector
  /// (num_groups_, or 1 with spatial_basis); sampling units (directions, or
  /// (direction, group) pairs).
  int64_t num_groups_ = 0;
  int64_t basis_groups_ = 0;
  int64_t num_units_ = 0;
  int64_t num_space_ = 0;
  int64_t ndim_ = 0;
  int64_t num_moments_ = 0;
  int64_t num_directions_ = 0;
  int64_t num_sets_ = 0;
  int64_t rank_ = 0;

  /// (r, basis_groups, num_space) basis (rows of storage).
  Growing basis_;
  /// (r, r) collision projections (one; per group with spatial_basis) and
  /// per dimension streaming projections.
  std::vector<Growing> collision_;
  std::vector<Growing> streaming_;
  /// X^T s_m: (num_moments, r), or (num_moments num_groups, r) per (m, g)
  /// with spatial_basis.
  Growing source_;
  std::vector<FaceProjection> faces_;
  /// Volume operators of the whole fiber, applied in one pass: the units'
  /// pair lists offset into the fiber (P), their mass and streaming values
  /// ((P) and (ndim, P)), and (num_groups, num_space) sigma_t per group of
  /// the set and dof.
  torch::Tensor volume_row_;
  torch::Tensor volume_col_;
  torch::Tensor volume_mass_;
  torch::Tensor volume_grad_;
  torch::Tensor sigma_;
  /// The sources at the set's groups, (num_moments, num_groups,
  /// num_space), gathered at the first extend() after reset().
  mutable torch::Tensor source_all_;
  /// (rows, r) every face's traces (FaceProjection::row0), and how to form
  /// them from the flattened (num_groups num_space) basis vectors: per row
  /// the (rows, A) indices and values (zero-padded to the widest face).
  /// Flat faces come first, grouped by normal, then curved ones.
  Growing traces_own_;
  Growing traces_upwind_;
  torch::Tensor own_index_;
  torch::Tensor own_value_;
  torch::Tensor upwind_index_;
  torch::Tensor upwind_value_;
  int64_t num_trace_rows_ = 0;
  bool any_internal_ = false;
  /// Per trace row its point's unnormalized normal n_q (rows, ndim) and
  /// quadrature weight w_q (rows), and per angle set 1 on the rows of the
  /// faces the set sweeps upwind first (num_sets, rows): the per-point
  /// upwind weights (Omega . n_q)_+- w_q of residuals().
  torch::Tensor row_normal_;
  torch::Tensor row_weight_;
  torch::Tensor row_dependency_;
  /// Per flat normal its rows [start, start + count) of the traces, the
  /// weights |n_q| w_q of those rows, and per angle set the same weights on
  /// the faces the set sweeps upwind first (DEPENDENCY), zero elsewhere
  /// (undefined if none).
  std::vector<std::pair<int64_t, int64_t>> normal_rows_;
  std::vector<torch::Tensor> normal_measure_;
  std::vector<std::vector<torch::Tensor>> dependency_measure_;
  /// Distinct unit normals of the flat faces, (ndim) each, their outflow
  /// projections F_n, and per angle set their dependency inflow K_{a,n}.
  std::vector<torch::Tensor> normals_;
  std::vector<Growing> outflow_;
  std::vector<std::vector<Growing>> inflow_;
  /// Curved faces (indices into faces_); those projected per sign class,
  /// with the capacity of their matrices; the others (indices into faces_),
  /// and per direction their cached projection D_d (bordered up to its own
  /// size).
  std::vector<size_t> curved_;
  std::vector<FaceClasses> classes_;
  int64_t classes_capacity_ = 0;
  std::vector<size_t> per_direction_;
  std::vector<Growing> curved_cache_;
  /// Per angle set, the (unit, face) pairs whose inflow comes from a buffer
  /// (lagged edges, reflective faces).
  std::vector<std::vector<std::pair<size_t, size_t>>> buffered_;
  /// Options::scattering: Legendre orders L (the solver's), the order of
  /// each moment, per local unit the within-set transfer (U, L, G, G)
  /// (rows the destination group, the set's groups only), whether every
  /// unit's is symmetric (then S_l is, and K_l needs no new rows), and the
  /// projections K_l (r, r).
  int64_t num_orders_ = 0;
  std::vector<int64_t> moment_order_;
  torch::Tensor scatter_xs_;
  bool scatter_symmetric_ = true;
  std::vector<Growing> scatter_;

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

  /// @brief The volume pair lists of the whole fiber and sigma_t per dof.
  void build_volume();
  /// @brief Gather the sources at the set's groups (source_all_) if not
  /// yet done in this transport solve.
  void gather_sources() const;
  /// @brief Add unit u's buffered inflow (lagged edges, reflective faces:
  /// data, not unknown) to out (basis_groups, num_space), assembled by the
  /// sweep kernel's buffered_inflow() as a sweep adds it.
  void add_buffered_inflow(int64_t u, double* out) const;
  /// @return Unit u's direction, and its group (index into groups_; -1:
  /// every group of the set, direction sampling).
  int64_t unit_direction(int64_t u) const noexcept
  {
    return options_.spatial_basis ? u / num_groups_ : u;
  }
  int64_t unit_group(int64_t u) const noexcept
  {
    return options_.spatial_basis ? u % num_groups_ : -1;
  }
  /// @return What a unit is, for messages ("direction" or "unit").
  std::string unit_name(bool capital) const
  {
    return options_.spatial_basis ? (capital ? "Unit" : "unit")
                                  : (capital ? "Direction" : "direction");
  }
  /// @brief Add the within-set transfer of order l to out (G, Ns), dof by
  /// dof: out_g += sum_g' sigma_{s,l,u}(g <- g') y_g' (transpose: sigma(g'
  /// <- g)), y already multiplied by the mass matrix (sigma is constant on a
  /// unit, so it commutes with M_u).
  void add_transfer(
    const double* y, int64_t l, bool transpose, double* out) const;
  /// @brief Assemble and LU-factor the reduced systems of `units`, in
  /// parallel over units, and call fn(i, lu, ipiv, b) for units[i] (lu the
  /// row-major factors for linalg::dense::lu_solve(), b its X^T b_u). The
  /// per-direction curved caches are bordered first, once per direction.
  /// @throws std::runtime_error On a unit out of range or repeated, or a
  /// singular reduced matrix (named by `caller`).
  void factor_units(const std::vector<int64_t>& units, const char* caller,
    const std::function<void(
      int64_t, const double*, const int64_t*, const torch::Tensor&)>& fn);
  /// @brief The trace rows of every face (FaceProjection::row0), their
  /// gather indices and values, and the per-normal row blocks and weights.
  void build_traces();
  /// @brief Border D_d (the per-direction curved faces) up to the current
  /// rank.
  void update_curved(int64_t d);
  /// @brief Border the sign-class matrices of the curved faces with the
  /// new columns J = [r0, r1) (traces already extended).
  void extend_classes(int64_t r0, int64_t r1);
  /// @return (basis_groups Qf, r) a face's own traces (rows g * Qf + q).
  torch::Tensor face_own(const FaceProjection& fp) const
  {
    return traces_own_.view().narrow(0, fp.row0, fp.rows);
  }
  /// @return (basis_groups Qf, r) an INTERNAL face's upwind traces.
  torch::Tensor face_upwind(const FaceProjection& fp) const
  {
    return traces_upwind_.view().narrow(0, fp.row0, fp.rows);
  }

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new ProjectedTransport(std::forward<Args>(args)...));
  }

  /// @brief Drop the basis (a new transport solve: new sources and traces).
  void reset() override final;

  /// @brief Append basis vectors and border every projection.
  /// @param X (p, basis_groups, fiber_length) new basis vectors (unweighted,
  /// the set's groups in order, or one spatial flux with spatial_basis).
  /// @throws std::runtime_error On a wrong shape.
  void extend(const torch::Tensor& X) override final;

  /// @brief Assemble and solve the reduced systems of `units` (distinct):
  /// an LU (LAPACK dgetrf/dgetrs, single-threaded) per unit, in parallel
  /// over units (the per-direction curved caches bordered first, once per
  /// direction).
  /// @return (U, r) coefficients c_u.
  /// @throws std::runtime_error On a unit out of range or repeated, or a
  /// singular reduced matrix (a rank-deficient basis; with sigma_t > 0 the
  /// symmetric part of the upwind operator is positive definite, so a basis
  /// of independent vectors gives a nonsingular one).
  torch::Tensor solve(const std::vector<int64_t>& units) override final;

  /// @brief Residuals b_u - A_u X c_u over every local unit, all units
  /// at once and matrix-free on the whole fiber: Y = C X (one GEMM), the
  /// volume terms through the fiber's pair lists, the face terms through
  /// the global traces with per-point upwind weights, b_d the sources plus
  /// the buffered inflow (the kernel's buffered_inflow()). Equal to
  /// SweepSolver::transport_residual() of X c_d (unless implicit_lagged).
  /// @param C (U, r) coefficients.
  /// @return (U, basis_groups, fiber_length) residuals.
  /// @throws std::runtime_error On a wrong shape or a unit out of range.
  torch::Tensor residuals(const std::vector<int64_t>& units,
    const torch::Tensor& C) const override final;

  /// @brief The reduced system of unit u, X^T A_u X and X^T b_u. Units of
  /// different directions may be assembled concurrently (each borders only
  /// its direction's cache).
  /// @return (r, r) matrix and (r) right-hand side.
  /// @throws std::runtime_error If u is out of range.
  std::pair<torch::Tensor, torch::Tensor> reduced_system(int64_t u);

  /// @brief The solver's current flux moments at the set's groups, (M, G,
  /// Ns) in the fiber layout: during a transport solve under source
  /// iteration, the moments that made its within-set scattering source.
  torch::Tensor solver_moments() const;

  /// @brief X^T s_m, the projected source moments of the running transport
  /// solve (scattering included), (M, r). Direction sampling.
  torch::Tensor source_projection() const;

  /// @brief Projected within-set scattering of a flux: row m is X^T S_{l(m)}
  /// phi_m (zero for l(m) >= L). Options::scattering.
  /// @param phi (M, G, Ns) moments at the set's groups, fiber layout.
  /// @return (M, r).
  /// @throws std::runtime_error Without Options::scattering, or on a wrong
  /// shape.
  torch::Tensor project_scattering(const torch::Tensor& phi) const;

  /// @return (r, r) K_l = X^T S_l X. Options::scattering.
  /// @throws std::runtime_error Without it, or if l is out of range.
  torch::Tensor scattering(int64_t l) const;

  /// @brief Han et al. (2026)'s angular Galerkin problem (eq. 3.4, with
  /// angular moments): the flux psi_d = X L_d of every direction, coupled
  /// through its own within-set scattering,
  ///   A_d L_d - sum_m Y_m(d) K_{l(m)} Lam_m = f_d,
  ///   Lam_m = sum_d w_d Y_m(d) L_d,
  ///   f_d = sum_m Y_m(d) F_m + X^T (buffered inflow of d),
  /// with A_d = X^T L_d X the reduced transport matrix. Eliminating L_d,
  ///   (I - T) Lam = sum_d w_d Y(d) (x) A_d^{-1} f_d,
  ///   T_{m,m'} = [sum_d w_d Y_m(d) Y_m'(d) A_d^{-1}] K_{l(m')},
  /// one (M r)-system; then L_d = A_d^{-1} (f_d + sum_m Y_m(d) K_{l(m)}
  /// Lam_m). One LU per direction (the first pass, which also forms
  /// A_d^{-1}, keeps the factors: D r^2 doubles), the second pass only
  /// solves with them; in parallel over directions in fixed blocks, so the
  /// result does not depend on the thread count.
  /// @param F (M, r) the fixed source moments, projected: X^T of the source
  /// without the within-set scattering (e.g. source_projection() -
  /// project_scattering(solver_moments())).
  /// @return L (D, r) and Lam (M, r).
  /// @throws std::runtime_error Without Options::scattering, on a wrong
  /// shape, or a singular system.
  std::pair<torch::Tensor, torch::Tensor> angular_galerkin(
    const torch::Tensor& F) override final;

  /// @brief source_projection() - project_scattering(solver_moments()): the
  /// projected source without its within-set scattering, during a
  /// Richardson transport solve (Options::scattering).
  /// @return (M, r).
  torch::Tensor fixed_source() const override final
  {
    return source_projection() - project_scattering(solver_moments());
  }

  /// @brief The part of X^T b_u that is data rather than source: the
  /// inflow of unit u read from the buffers (lagged internal edges of its
  /// direction's angle set, reflective faces), projected.
  /// @return (r) X^T (lagged and reflected inflow of unit u).
  /// @throws std::runtime_error If u is out of range.
  torch::Tensor buffered_rhs(int64_t u) const;

  // =================================================================
  // Public getters
  int64_t rank() const noexcept override final { return rank_; }
  int64_t num_directions() const noexcept override final
  {
    return num_directions_;
  }
  int64_t num_units() const noexcept override final { return num_units_; }
  int64_t basis_groups() const noexcept override final { return basis_groups_; }
  /// @return (r, basis_groups, fiber_length) the basis.
  torch::Tensor basis() const override final;
  const std::vector<int64_t>& groups() const noexcept { return groups_; }
  /// @return Legendre orders of the projected scattering (0 without
  /// Options::scattering).
  int64_t num_orders() const noexcept { return num_orders_; }
  /// @return Distinct unit normals of the flat faces.
  int64_t num_normals() const noexcept
  {
    return static_cast<int64_t>(normals_.size());
  }
  /// @return Faces projected by quadrature (curved), per sign class or per
  /// direction.
  int64_t num_curved_faces() const noexcept
  {
    return static_cast<int64_t>(curved_.size());
  }
  /// @return Per curved face (in curved_ order) its number of sign classes,
  /// or -1 if it is projected per direction.
  std::vector<int64_t> face_classes() const;
};

} // namespace ttnte::sweep
