#pragma once

#include "ttnte/math/quadrature_set.hpp"
#include <torch/extension.h>
#include <utility>

namespace ttnte::sweep {

/// @brief Discrete ordinates of a quadrature set as streaming directions.
/// @param qset The angular quadrature (1-D Gauss-Legendre for slabs, a
/// product Gauss-Legendre-Chebyshev set otherwise).
/// @param ndim Number of spatial dimensions.
/// @return (Omega, Omega3, weights): Omega is (N_Omega, ndim) and holds the
/// components that stream in this geometry; Omega3 is the full (N_Omega, 3)
/// unit vector (used to match reflected directions); weights is
/// (N_Omega). Directions are flattened in the TT core order (polar major,
/// azimuthal minor).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> discrete_ordinates(
  const math::QuadratureSet& qset, int64_t ndim);

/// @brief Real spherical harmonics evaluated at the discrete ordinates, with
/// the normalization of the TT scattering kernel
/// (DGFirstOrderTransportBackend::assemble_scattering_kernel), so that
///   K_l(d, d') = sum_{m in l} Y_m(d) Y_m(d')
/// and the scattering source is
///   q_{d,g} = sum_m Y_m(d) sum_g' sigma_{l(m), g <- g'} M phi_{m, g'},
///   phi_m = sum_d w_d Y_m(d) psi_d.
/// 1-D uses sqrt(2l + 1) P_l(mu). 2-D folds the sphere onto mu > 0 and keeps
/// only the m with the parity of l. 3-D keeps every m.
/// @param qset The angular quadrature.
/// @param ndim Number of spatial dimensions.
/// @param num_orders Number of Legendre orders L (scattering expansion size).
/// @return (Y, order): Y is (N_Omega, num_moments) and order is
/// (num_moments) int64 holding l for each moment.
std::pair<torch::Tensor, torch::Tensor> discrete_harmonics(
  const math::QuadratureSet& qset, int64_t ndim, int64_t num_orders);

} // namespace ttnte::sweep
