#pragma once

#include "ttnte/cad/patch.hpp"
#include "ttnte/mesh/mesh_block_boundary.hpp"
#include <cstdint>
#include <torch/extension.h>
#include <vector>

namespace ttnte::cad {

/// @brief Quadrature shared by the two faces of an interface, as points in
/// each face's own parametric coordinates (same point order on both sides).
struct InterfaceQuadrature {
  /// (Q, ndim - 1) points in the first face's tangential parameters
  /// (natural order, the face dimension removed).
  torch::Tensor u_first;
  /// (Q, ndim - 1) the same points in the second face's tangential
  /// parameters.
  torch::Tensor u_second;
  /// (Q) parametric measure in the first face's parameters (span width
  /// times Gauss-Legendre weight), so that with the first face's scaled
  /// normal n_q, sum_q f_q |n_q| w_q is the surface integral of f.
  torch::Tensor weights;
  /// The same rule per tangential dimension k of the first face (the
  /// points above are their tensor grid, C order over k): the points in the
  /// first face's parameter, the same points in the second face's
  /// parameter along second_dim[k], and the first face's 1-D measure.
  std::vector<torch::Tensor> points_first;
  std::vector<torch::Tensor> points_second;
  std::vector<torch::Tensor> weights_first;
  /// Tangential dimension of the second face (index among its tangential
  /// dimensions) along each tangential dimension of the first, and whether
  /// it runs backwards.
  std::vector<int64_t> second_dim;
  std::vector<bool> flip;
};

/// @brief Build the quadrature of an interface from the two faces'
/// tangential bases.
/// @param first Tangential bases of the first face (natural order).
/// @param second Tangential bases of the second face (natural order).
/// @param second_dim Index into `second` of the basis along each `first`.
/// @param flip Whether each `first` dimension runs backwards on the second.
/// @return The shared quadrature (see the overload on patches).
InterfaceQuadrature interface_quadrature(const std::vector<BSplineBasis>& first,
  const std::vector<BSplineBasis>& second,
  const std::vector<int64_t>& second_dim, const std::vector<bool>& flip);

/// @brief Build the quadrature of an interface between two faces that are
/// the same surface refined differently (whole face to whole face, no
/// T-junctions). Along each tangential parameter the sub-spans are the
/// union of both sides' knot spans and carry max(p_first, p_second) + 1
/// Gauss-Legendre points, so the integrals of products of the two sides'
/// basis functions are exact on affine faces. Both sides of the interface
/// integrate their fluxes with these points so the outflow of one side is
/// the inflow of the other to round-off (particle balance).
/// @param first The patch owning the first face.
/// @param dim_first Parametric dimension the first face is normal to.
/// @param second The patch owning the second face.
/// @param dim_second Parametric dimension the second face is normal to.
/// @param mapping How the first face reads data of the second (the
/// NeighborInfo::mapping of the first face's connection): perm[i] is the
/// second patch's dimension along the first patch's dimension i, and
/// flip[k] reverses the k-th tangential dimension of the first face.
/// @return The shared quadrature.
InterfaceQuadrature interface_quadrature(const Patch& first, int64_t dim_first,
  const Patch& second, int64_t dim_second,
  const mesh::BoundaryMapping& mapping);

} // namespace ttnte::cad
