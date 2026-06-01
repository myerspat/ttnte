#include "ttnte/math/special.hpp"
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <torch/extension.h>
#include <torch/types.h>

namespace ttnte::math::special {

torch::Tensor legendre(uint32_t l, const torch::Tensor& x, bool skip_checks)
{
  if (!skip_checks) {
    TORCH_CHECK(
      x.device() == torch::kCPU, "This function is only supported on CPU");
    TORCH_CHECK(x.ndimension() == 1, "`x` must be a 1-D tensor");
  }

  // Enforce continuity
  auto x_c = x.contiguous();

  return AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "legendre_cpu", [&] {
    torch::Tensor p = torch::empty_like(x_c, x_c.options());

    // Get the accessors
    auto x_acc = x_c.accessor<scalar_t, 1>();
    auto p_acc = p.accessor<scalar_t, 1>();

    // Iterate through the points given
    at::parallel_for(0, x_c.size(0), 0, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        p_acc[i] = std::legendre(l, x_acc[i]);
      }
    });

    return p;
  });
}

torch::Tensor assoc_legendre(
  uint32_t l, uint32_t m, const torch::Tensor& x, bool skip_checks)
{
  if (!skip_checks) {
    TORCH_CHECK(m <= l, "`m` was too large, expected `0 <= m <= l`");
    TORCH_CHECK(
      x.device() == torch::kCPU, "This function is only supported on CPU");
    TORCH_CHECK(x.ndimension() == 1, "`x` must be a 1-D tensor");
  }

  // Enforce continuity
  auto x_c = x.contiguous();

  return AT_DISPATCH_FLOATING_TYPES(
    x_c.scalar_type(), "assoc_legendre_cpu", [&] {
      torch::Tensor p = torch::empty_like(x_c, x_c.options());

      // Get the accessors
      auto x_acc = x_c.accessor<scalar_t, 1>();
      auto p_acc = p.accessor<scalar_t, 1>();

      // Iterate through the points given
      at::parallel_for(0, x_c.size(0), 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
          p_acc[i] = std::assoc_legendre(l, m, x_acc[i]);
        }
      });

      return p;
    });
}

torch::Tensor assoc_legendre(const torch::Tensor& l, const torch::Tensor& m,
  const torch::Tensor& x, bool skip_checks)
{
  if (!skip_checks) {
    TORCH_CHECK(l.device() == torch::kCPU && m.device() == torch::kCPU &&
                  x.device() == torch::kCPU,
      "This function is only supported on CPU");
    TORCH_CHECK(
      l.ndimension() == 1 && m.ndimension() == 1 && x.ndimension() == 1,
      "`l`, `m`, and `x` must be 1-D tensors");
    TORCH_CHECK(x.is_floating_point(), "`x` must be a floating point tensor");
    TORCH_CHECK(l.numel() == x.numel() && m.numel() == x.numel(),
      "`l`, `m`, and `x` must be the same length");
    TORCH_CHECK((m >= 0).all().item<bool>(), "`m` should not be negative");
    TORCH_CHECK((l >= 0).all().item<bool>(), "`l` should not be negative");
    TORCH_CHECK(
      (m <= l).all().item<bool>(), "`m` was too large, expected `0 <= m <= l`");
  }

  // Enforce continuity
  auto l_c = l.to(torch::kUInt32, true, false, at::MemoryFormat::Contiguous);
  auto m_c = m.to(torch::kUInt32, true, false, at::MemoryFormat::Contiguous);
  auto x_c = x.contiguous();

  // Accessors
  auto l_acc = l_c.accessor<uint32_t, 1>();
  auto m_acc = m_c.accessor<uint32_t, 1>();

  return AT_DISPATCH_FLOATING_TYPES(
    x_c.scalar_type(), "assoc_legendre_cpu", [&] {
      auto p = torch::empty_like(x_c, x_c.options());

      // Get the accessors
      auto x_acc = x_c.accessor<scalar_t, 1>();
      auto p_acc = p.accessor<scalar_t, 1>();

      // Iterate through the points given
      at::parallel_for(0, x_c.size(0), 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
          p_acc[i] = std::assoc_legendre(l_acc[i], m_acc[i], x_acc[i]);
        }
      });

      return p;
    });
}

torch::Tensor sph_harm(uint32_t l, uint32_t m, const torch::Tensor& mu,
  const torch::Tensor& gamma, HarmonicComponent component, bool skip_checks)
{
  if (!skip_checks) {
    TORCH_CHECK(m <= l, "`m` was too large, expected `0 <= m <= l`");
    TORCH_CHECK(torch::logical_and(mu >= -1, mu <= 1).all().item<bool>(),
      "The range of `mu` must be `-1 <= mu <= 1`");
    TORCH_CHECK(torch::logical_and(gamma >= 0, gamma <= 2 * std::numbers::pi)
                  .all()
                  .item<bool>(),
      "The range of `gamma` must be `0 <= gamma <= 2 * pi`");
    TORCH_CHECK(mu.device() == torch::kCPU && gamma.device() == torch::kCPU,
      "This function is only supported on CPU");
    TORCH_CHECK(mu.is_floating_point() && gamma.is_floating_point(),
      "`mu` and `gamma` must be floating point tensors");
    TORCH_CHECK(mu.scalar_type() == gamma.scalar_type(),
      "`mu` and `gamma` must have the same scalar type");
    TORCH_CHECK(mu.ndimension() == 1 && gamma.ndimension() == 1,
      "`mu` and `gamma` must be 1-D tensors");
    TORCH_CHECK(mu.numel() == gamma.numel(),
      "`mu` and `gamma` must have the same length");
  }

  // Enforce continuity
  auto mu_c = mu.contiguous();
  auto gamma_c = gamma.contiguous();

  return AT_DISPATCH_FLOATING_TYPES(mu.scalar_type(), "sph_harm_cpu", [&] {
    int64_t num_points = mu.size(0);

    // Cast integers to float
    scalar_t one = 1.0;
    scalar_t two = 2.0;
    scalar_t l_f = static_cast<scalar_t>(l);
    scalar_t m_f = static_cast<scalar_t>(m);

    auto y = torch::full(component == HarmonicComponent::BOTH
                           ? c10::SmallVector<int64_t, 2> {num_points, 2}
                           : c10::SmallVector<int64_t, 2> {num_points, 1},
      std::sqrt((two * l_f + one) * std::tgamma(l_f - m_f + one) /
                std::tgamma(l_f + m_f + one)),
      mu.options());

    // Accessors
    auto mu_acc = mu_c.accessor<scalar_t, 1>();
    auto gamma_acc = gamma_c.accessor<scalar_t, 1>();
    auto y_acc = y.accessor<scalar_t, 2>();

    // Iterate through the points given
    at::parallel_for(0, num_points, 0, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        // Evaluate associated Legendre polynomial
        scalar_t p = std::assoc_legendre(l, m, mu_acc[i]);
        int idx = 0;

        // Even term
        if (component != HarmonicComponent::ODD) {
          y_acc[i][idx] *= cos(m_f * gamma_acc[i]) * p;
          idx++;
        }

        // Odd term
        if (component != HarmonicComponent::EVEN) {
          y_acc[i][idx] *= sin(m_f * gamma_acc[i]) * p;
        }
      }
    });

    return y;
  });
}

torch::Tensor sph_harm(const torch::Tensor& l, const torch::Tensor& m,
  const torch::Tensor& mu, const torch::Tensor& gamma,
  HarmonicComponent component, bool skip_checks)
{
  if (!skip_checks) {
    TORCH_CHECK(l.device() == torch::kCPU && m.device() == torch::kCPU &&
                  mu.device() == torch::kCPU && gamma.device() == torch::kCPU,
      "This function is only supported on CPU");
    TORCH_CHECK(l.ndimension() == 1 && m.ndimension() == 1 &&
                  mu.ndimension() == 1 && gamma.ndimension() == 1,
      "`l`, `m`, `mu`, and `gamma` must be 1-D tensors");
    TORCH_CHECK(torch::logical_and(l >= 0, m >= 0).all().item<bool>(),
      "`l` and `m` must be positive integers");
    TORCH_CHECK(
      (m <= l).all().item<bool>(), "`m` was too large, expected `0 <= m <= l`");
    TORCH_CHECK(mu.is_floating_point() && gamma.is_floating_point(),
      "`mu` and `gamma` should be floating point tensors");
    TORCH_CHECK(l.numel() == m.numel() && l.numel() == mu.numel() &&
                  l.numel() == gamma.numel(),
      "`l`, `m`, `mu`, and `gamma` should all be the same length");
    TORCH_CHECK(torch::logical_and(mu >= -1, mu <= 1).all().item<bool>(),
      "The range of `mu` must be `-1 <= mu <= 1`");
    TORCH_CHECK(torch::logical_and(gamma >= 0, gamma <= 2 * std::numbers::pi)
                  .all()
                  .item<bool>(),
      "The range of `gamma` must be `0 <= gamma <= 2 * pi`");
    TORCH_CHECK(mu.scalar_type() == gamma.scalar_type(),
      "`mu` and `gamma` should have the same scalar type");
  }

  // Enforce continuity
  auto l_c = l.to(torch::kUInt32, true, false, at::MemoryFormat::Contiguous);
  auto m_c = m.to(torch::kUInt32, true, false, at::MemoryFormat::Contiguous);
  auto mu_c = mu.contiguous();
  auto gamma_c = gamma.contiguous();

  return AT_DISPATCH_FLOATING_TYPES(mu.scalar_type(), "sph_harm_cpu", [&] {
    int64_t num_points = mu.size(0);

    // Cast integers to float
    scalar_t one = 1.0;
    scalar_t two = 2.0;

    auto y = torch::empty(component == HarmonicComponent::BOTH
                            ? c10::SmallVector<int64_t, 2> {num_points, 2}
                            : c10::SmallVector<int64_t, 2> {num_points, 1},
      mu.options());

    // Accessors
    auto l_acc = l_c.accessor<uint32_t, 1>();
    auto m_acc = m_c.accessor<uint32_t, 1>();
    auto mu_acc = mu_c.accessor<scalar_t, 1>();
    auto gamma_acc = gamma_c.accessor<scalar_t, 1>();
    auto y_acc = y.accessor<scalar_t, 2>();

    // Iterate through the points given
    at::parallel_for(0, num_points, 0, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        scalar_t l_f = l_acc[i];
        scalar_t m_f = m_acc[i];
        scalar_t Cp =
          std::sqrt((two * l_f + one) * std::tgamma(l_f - m_f + one) /
                    std::tgamma(l_f + m_f + one)) *
          std::assoc_legendre(l_acc[i], m_acc[i], mu_acc[i]);
        int idx = 0;

        // Even term
        if (component != HarmonicComponent::ODD) {
          y_acc[i][idx] = Cp * cos(m_f * gamma_acc[i]);
          idx++;
        }

        // Odd term
        if (component != HarmonicComponent::EVEN) {
          y_acc[i][idx] = Cp * sin(m_f * gamma_acc[i]);
        }
      }
    });

    return y;
  });
}

torch::Tensor sph_harm(const torch::Tensor& l, const torch::Tensor& m,
  const torch::Tensor& mu, const torch::Tensor& gamma,
  const torch::Tensor& component, bool skip_checks)
{
  if (!skip_checks) {
    TORCH_CHECK(l.device() == torch::kCPU && m.device() == torch::kCPU &&
                  mu.device() == torch::kCPU && gamma.device() == torch::kCPU &&
                  component.device() == torch::kCPU,
      "This function is only supported on CPU");
    TORCH_CHECK(l.ndimension() == 1 && m.ndimension() == 1 &&
                  mu.ndimension() == 1 && gamma.ndimension() == 1 &&
                  component.ndimension() == 1,
      "`l`, `m`, `mu`, `gamma`, and `component` must be 1-D tensors");
    TORCH_CHECK(torch::logical_and(l >= 0, m >= 0).all().item<bool>(),
      "`l` and `m` must be positive integers");
    TORCH_CHECK(
      (m <= l).all().item<bool>(), "`m` was too large, expected `0 <= m <= l`");
    TORCH_CHECK(component.scalar_type() == torch::kBool,
      "Values in `component` must be Boolean with 0 (even) or 1 (odd)");
    TORCH_CHECK(mu.is_floating_point() && gamma.is_floating_point(),
      "`mu` and `gamma` should be floating point tensors");
    TORCH_CHECK(l.numel() == m.numel() && l.numel() == mu.numel() &&
                  l.numel() == gamma.numel() && l.numel() == component.numel(),
      "`l`, `m`, `mu`, `gamma`, and `component` should all be the same length");
    TORCH_CHECK(torch::logical_and(mu >= -1, mu <= 1).all().item<bool>(),
      "The range of `mu` must be `-1 <= mu <= 1`");
    TORCH_CHECK(torch::logical_and(gamma >= 0, gamma <= 2 * std::numbers::pi)
                  .all()
                  .item<bool>(),
      "The range of `gamma` must be `0 <= gamma <= 2 * pi`");
    TORCH_CHECK(mu.scalar_type() == gamma.scalar_type(),
      "`mu` and `gamma` should have the same scalar type");
  }

  // Enforce continuity
  auto l_c = l.to(torch::kUInt32, true, false, at::MemoryFormat::Contiguous);
  auto m_c = m.to(torch::kUInt32, true, false, at::MemoryFormat::Contiguous);
  auto mu_c = mu.contiguous();
  auto gamma_c = gamma.contiguous();
  auto component_c = component.contiguous();

  return AT_DISPATCH_FLOATING_TYPES(mu.scalar_type(), "sph_harm_cpu", [&] {
    int64_t num_points = mu.size(0);

    // Cast integers to float
    scalar_t one = 1.0;
    scalar_t two = 2.0;
    auto y = torch::empty({mu.numel()}, mu.options());

    // Accessors
    auto l_acc = l_c.accessor<uint32_t, 1>();
    auto m_acc = m_c.accessor<uint32_t, 1>();
    auto mu_acc = mu_c.accessor<scalar_t, 1>();
    auto gamma_acc = gamma_c.accessor<scalar_t, 1>();
    auto y_acc = y.accessor<scalar_t, 1>();
    auto component_acc = component_c.accessor<bool, 1>();

    // Iterate through the points given
    at::parallel_for(0, num_points, 0, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        scalar_t l_f = l_acc[i];
        scalar_t m_f = m_acc[i];
        scalar_t Cp =
          std::sqrt((two * l_f + one) * std::tgamma(l_f - m_f + one) /
                    std::tgamma(l_f + m_f + one)) *
          std::assoc_legendre(l_acc[i], m_acc[i], mu_acc[i]);

        if (component_acc[i]) {
          // Odd term
          y_acc[i] = Cp * sin(m_f * gamma_acc[i]);
        } else {
          // Even term
          y_acc[i] = Cp * cos(m_f * gamma_acc[i]);
        }
      }
    });

    return y;
  });
}

} // namespace ttnte::math::special
