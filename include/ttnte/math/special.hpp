#pragma once

#include <torch/extension.h>

namespace ttnte::math::special {

/// @brief Which spherical harmonics to compute.
enum class HarmonicComponent : uint8_t { EVEN, ODD, BOTH };

torch::Tensor legendre(
  uint32_t l, const torch::Tensor& x, bool skip_checks = false);

torch::Tensor assoc_legendre(
  uint32_t l, uint32_t m, const torch::Tensor& x, bool skip_checks = false);

torch::Tensor assoc_legendre(const torch::Tensor& l, const torch::Tensor& m,
  const torch::Tensor& x, bool skip_checks = false);

torch::Tensor sph_harm(uint32_t l, uint32_t m, const torch::Tensor& mu,
  const torch::Tensor& gamma,
  HarmonicComponent component = HarmonicComponent::BOTH,
  bool skip_checks = false);

torch::Tensor sph_harm(const torch::Tensor& l, const torch::Tensor& m,
  const torch::Tensor& mu, const torch::Tensor& gamma,
  HarmonicComponent component = HarmonicComponent::BOTH,
  bool skip_checks = false);

torch::Tensor sph_harm(const torch::Tensor& l, const torch::Tensor& m,
  const torch::Tensor& mu, const torch::Tensor& gamma,
  const torch::Tensor& component, bool skip_checks = false);

} // namespace ttnte::math::special
