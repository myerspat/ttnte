#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/python/numpy.hpp"
#include <numbers>

namespace ttnte::math {

void QuadratureSet::to_(
  const torch::Device& device, const torch::ScalarType& dtype)
{
  to_(torch::TensorOptions().device(device).dtype(dtype));
}

void QuadratureSet::to_(const torch::ScalarType& dtype)
{
  to_(torch::TensorOptions().dtype(dtype));
}

void QuadratureSet::to_(const torch::Device& device)
{
  to_(torch::TensorOptions().device(device));
}

QuadratureSet1D::Ptr QuadratureSet1D::gauss_legendre(
  int64_t n, double weighting_factor)
{
  const auto [points, weights] = python::numpy::Acquire().leggauss(n);
  return create(points, weights / weights.sum(), weighting_factor);
}

QuadratureSet1D::Ptr QuadratureSet1D::gauss_chebyshev(
  int64_t n, double weighting_factor)
{
  const auto options =
    at::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

  double one = 1.0;
  double two = 2.0;
  double four = 4.0;

  torch::Tensor points = (two * torch::arange(1, n + 1, options) - one) *
                         std::numbers::pi / (four * n);
  torch::Tensor weights = torch::ones({n}, options) / n;

  return create(points, weights / weights.sum(), weighting_factor);
}

void QuadratureSet1D::to_(const torch::TensorOptions& options)
{
  points_ = points_.to(options);
  weights_ = weights_.to(options);
}

ProductQuadrature::Ptr ProductQuadrature::gauss_legendre_chebyshev(
  int64_t n_polar, int64_t n_azimuthal, int64_t ndim, double weighting_factor)
{
  if (ndim < 2 || ndim > 3) {
    throw utils::runtime_error(
      "ttnte::math::ProductQuadrature::gauss_legendre_chebyshev",
      "`ndim` must be 2 or 3");
  }

  // Constants
  double one = 1.0;
  double two = 2.0;
  double pi = std::numbers::pi;

  // Quadrature sets along each dimension for integrating a sphere
  auto polar_qset = QuadratureSet1D::gauss_legendre(n_polar, 1.0);
  auto azimuthal_qset = QuadratureSet1D::gauss_chebyshev(n_azimuthal, 1.0);

  // Wrap the azimuthal one around the full 4pi
  const auto& az_points = azimuthal_qset->get_points();
  const auto& az_weights = azimuthal_qset->get_weights();

  auto options = az_points.options();
  auto full_az_points = torch::empty(4 * n_azimuthal, options);
  auto full_az_weights = torch::empty(4 * n_azimuthal, options);

  for (int64_t i = 0; i < 4; i++) {
    full_az_points.narrow(0, i * n_azimuthal, n_azimuthal)
      .copy_(static_cast<double>(i) * pi / two + az_points);
    full_az_weights.narrow(0, i * n_azimuthal, n_azimuthal).copy_(az_weights);
  }

  full_az_weights /= full_az_weights.sum();
  azimuthal_qset = QuadratureSet1D::create(
    std::move(full_az_points), std::move(full_az_weights), 1.0);

  // Reflect the polar points across the xy-plane
  auto po_points = polar_qset->get_points();
  auto po_weights = polar_qset->get_weights();
  po_points.add_(one).div_(two);

  if (ndim == 3) {
    // Reflect the polar points across the xy-plane
    auto full_po_points = torch::cat({-torch::flip(po_points, {0}), po_points});
    auto full_po_weights =
      torch::cat({torch::flip(po_weights, {0}), po_weights});
    full_po_weights /= full_po_weights.sum();

    polar_qset = QuadratureSet1D::create(
      std::move(full_po_points), std::move(full_po_weights), 1.0);

  } else {
    polar_qset =
      QuadratureSet1D::create(std::move(po_points), std::move(po_weights), 1.0);
  }

  return create(Quads {std::move(polar_qset), std::move(azimuthal_qset)},
    weighting_factor, Symmetry::XY_PLANE);
}

c10::SmallVector<torch::Tensor, 3> ProductQuadrature::get_factored_points()
  const noexcept
{
  c10::SmallVector<torch::Tensor, 3> points;
  points.reserve(get_ndim());
  for (const auto& quad : quads_) {
    points.push_back(quad->get_points());
  }
  return points;
}

c10::SmallVector<torch::Tensor, 3> ProductQuadrature::get_factored_weights()
  const noexcept
{
  c10::SmallVector<torch::Tensor, 3> weights;
  weights.reserve(get_ndim());
  for (const auto& quad : quads_) {
    weights.push_back(quad->get_weights());
  }
  return weights;
}

torch::Tensor ProductQuadrature::get_points() const
{
  return torch::stack(torch::meshgrid(get_factored_points(), "ij"), -1)
    .reshape({-1, get_ndim()});
}

torch::Tensor ProductQuadrature::get_weights() const
{
  int64_t ndim = get_ndim();

  // Compute the weights through broadcasting
  torch::Tensor weights;
  for (size_t i = 0; i < ndim; i++) {
    const auto& quad = quads_[i];

    c10::SmallVector<int64_t, 3> shape(ndim, 1);
    shape[i] = quad->get_num_dofs();

    weights = i == 0 ? quad->get_weights().reshape(shape)
                     : weights * quad->get_weights().reshape(shape);
  }

  return weights.flatten();
}

int64_t ProductQuadrature::get_num_dofs() const
{
  int64_t num_dofs = 1;
  for (const auto& quad : quads_) {
    num_dofs *= quad->get_num_dofs();
  }
  return num_dofs;
}

void ProductQuadrature::to_(const torch::TensorOptions& options)
{
  for (auto& quad : quads_) {
    quad->to_(options);
  }
}

} // namespace ttnte::math
