#include "ttnte/sweep/angular_moments.hpp"
#include "ttnte/math/special.hpp"
#include "ttnte/utils/exception.hpp"
#include <cmath>
#include <vector>

namespace {

/// @return (mu, gamma, weights) flattened over (polar, azimuthal) for a
/// product quadrature.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> product_points(
  const ttnte::math::QuadratureSet& qset)
{
  if (!qset.is_tensor_product()) {
    throw ttnte::utils::runtime_error("ttnte::sweep::discrete_ordinates",
      "Multi-dimensional sweeps require a product quadrature set");
  }
  const auto& pq = static_cast<const ttnte::math::ProductQuadrature&>(qset);
  auto points = pq.get_factored_points();
  auto weights = pq.get_factored_weights();
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  auto mu = points[0].to(opts);
  auto gamma = points[1].to(opts);
  auto w = torch::outer(weights[0].to(opts), weights[1].to(opts)).flatten();
  return {mu, gamma, w};
}

} // namespace

namespace ttnte::sweep {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> discrete_ordinates(
  const math::QuadratureSet& qset, int64_t ndim)
{
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  if (ndim == 1) {
    if (qset.is_tensor_product()) {
      throw utils::runtime_error("ttnte::sweep::discrete_ordinates",
        "Slab sweeps require a 1-D quadrature set");
    }
    auto mu = qset.get_points().to(opts).flatten();
    auto w = qset.get_weights().to(opts).flatten();
    auto omega3 =
      torch::stack({mu, torch::zeros_like(mu), torch::zeros_like(mu)}, 1);
    return {mu.unsqueeze(1), omega3, w};
  }

  auto [mu, gamma, w] = product_points(qset);
  int64_t nm = mu.size(0);
  int64_t ng = gamma.size(0);
  auto sin_theta = torch::sqrt(1.0 - mu.square()).unsqueeze(1).expand({nm, ng});
  auto ox = (sin_theta * torch::cos(gamma).unsqueeze(0)).flatten();
  auto oy = (sin_theta * torch::sin(gamma).unsqueeze(0)).flatten();
  auto oz = mu.unsqueeze(1).expand({nm, ng}).flatten();
  auto omega3 = torch::stack({ox, oy, oz}, 1);
  return {omega3.narrow(1, 0, ndim).contiguous(), omega3, w};
}

std::pair<torch::Tensor, torch::Tensor> discrete_harmonics(
  const math::QuadratureSet& qset, int64_t ndim, int64_t num_orders)
{
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  std::vector<torch::Tensor> columns;
  std::vector<int64_t> orders;

  if (ndim == 1) {
    auto mu = qset.get_points().to(opts).flatten();
    for (int64_t l = 0; l < num_orders; l++) {
      columns.push_back(l == 0 ? torch::ones_like(mu)
                               : std::sqrt(2.0 * l + 1.0) *
                                   math::special::legendre(l, mu, false));
      orders.push_back(l);
    }
  } else {
    auto [mu, gamma, w] = product_points(qset);
    int64_t nm = mu.size(0);
    int64_t ng = gamma.size(0);
    for (int64_t l = 0; l < num_orders; l++) {
      if (l == 0) {
        columns.push_back(torch::ones({nm * ng}, opts));
        orders.push_back(0);
        continue;
      }
      // 2-D: the half-sphere fold keeps only m with the parity of l (see
      // assemble_scattering_kernel()); 3-D integrates the full sphere.
      const int64_t m_start = ndim == 2 ? l % 2 : 0;
      const int64_t m_step = ndim == 2 ? 2 : 1;
      for (int64_t m = m_start; m <= l; m += m_step) {
        double log_factor =
          0.5 * (std::log(2.0 * l + 1.0) + std::lgamma(l - m + 1.0) -
                  std::lgamma(l + m + 1.0));
        double scale = (m != 0 ? std::sqrt(2.0) : 1.0) * std::exp(log_factor);
        auto plm = scale * math::special::assoc_legendre(l, m, mu, false);
        columns.push_back(
          (plm.unsqueeze(1) * torch::cos(m * gamma).unsqueeze(0)).flatten());
        orders.push_back(l);
        if (m != 0) {
          columns.push_back(
            (plm.unsqueeze(1) * torch::sin(m * gamma).unsqueeze(0)).flatten());
          orders.push_back(l);
        }
      }
    }
  }

  auto Y = torch::stack(columns, 1).contiguous();
  auto order = torch::tensor(orders, torch::kInt64);
  return {Y, order};
}

} // namespace ttnte::sweep
