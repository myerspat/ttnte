#pragma once

#include "ttnte/sweep/patch_discretization.hpp"
#include <memory>
#include <vector>

namespace ttnte::sweep {

/// @brief Matrix-free application of a patch's local operator
///   L(Omega, sigma) u = sigma M u - sum_i Omega_i G_i u + sum_f B_f^+ u
/// by sum factorization (Antolin et al., CMAME 285, 2015; Bressan and
/// Takacs, CMAME 352, 2019): the volume quadrature is a tensor grid and the
/// B-splines are tensor products, so values at the quadrature points and the
/// testing against basis functions are mode products with the sparse 1-D
/// basis matrices, O(N (p + 1)) each. Curved (rational) geometry enters only
/// through pointwise data: the weight function W and its gradient, |det J|
/// and J^{-1}. The outflow faces use the face pair lists.
class SumFactorization {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<SumFactorization>;

  /// @brief Scratch of one worker.
  /// Scratch of one apply(), kept between calls: the two ping-pong buffers
  /// of the mode products, the pointwise fields at the quadrature points,
  /// the current tensor shape, and one face's outflow weights.
  struct Workspace {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> fields;
    std::vector<int64_t> shape;
    std::vector<double> outflow;
  };

private:
  // =================================================================
  // Private data
  const PatchDiscretization* disc_ = nullptr;
  int64_t ndim_ = 0;
  std::vector<int64_t> n_;
  std::vector<int64_t> nq_;
  std::vector<int64_t> p1_;
  int64_t num_dofs_ = 0;
  int64_t num_points_ = 0;
  /// Control weights (empty if not rational), and on the quadrature grid
  /// W, its parametric gradient (Q, ndim), |det J| w and J^{-1}.
  std::vector<double> control_weights_;
  std::vector<double> W_;
  std::vector<double> dW_;
  std::vector<double> measure_;
  std::vector<double> inverse_jacobian_;

  // =================================================================
  // Private constructors (see the static factories)
  explicit SumFactorization(const PatchDiscretization& disc);

  /// @brief Mode product along `dim`: dofs -> quadrature points
  /// (forward) or the transpose (accumulated into out).
  void forward(int64_t dim, bool derivative, const std::vector<int64_t>& shape,
    const double* in, double* out) const;
  void transpose(int64_t dim, bool derivative,
    const std::vector<int64_t>& shape, const double* in, double* out) const;

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new SumFactorization(std::forward<Args>(args)...));
  }

  /// @brief y = L(omega, sigma) u.
  void apply(const double* omega, double sigma, const double* u, double* y,
    Workspace& work) const;

  int64_t num_dofs() const noexcept { return num_dofs_; }
};

} // namespace ttnte::sweep
