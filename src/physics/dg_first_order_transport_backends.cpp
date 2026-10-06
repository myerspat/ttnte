#include "ttnte/physics/dg_first_order_transport_backends.hpp"
#include "ttnte/linalg/tt_ops.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/math/special.hpp"
#include "ttnte/utils/exception.hpp"
#include <array>
#include <c10/core/ScalarType.h>
#include <cstdlib>

namespace {

template<typename Func>
ttnte::linalg::TTEngine apply_mult_separable(const ttnte::linalg::TTEngine& tt,
  Func func, const ttnte::linalg::TTConfig& rounding,
  const ttnte::linalg::CrossConfig& cross, int max_dense_size)
{
  static_assert(
    std::is_invocable_r_v<torch::Tensor, Func, const torch::Tensor&>,
    "FATAL ERROR: The provided function must take a single 'const\n"
    "torch::Tensor&' and return a 'torch::Tensor'.");

  c10::SmallVector<int64_t, 6> m_modes;
  c10::SmallVector<int64_t, 6> n_modes;
  n_modes.reserve(tt.size());
  m_modes.reserve(tt.size());
  int64_t size = 1;
  bool is_rank_one = true;
  for (const auto& core : tt) {
    m_modes.push_back(core.size(1));
    n_modes.push_back(core.size(2));
    size *= m_modes.back() * n_modes.back();

    if (is_rank_one && core.size(3) != 1) {
      is_rank_one = false;
    }
  }

  // TT is rank one we can just compute the operation for individual tensors
  if (is_rank_one) {
    ttnte::linalg::TTEngine::Tensors result;
    result.reserve(tt.size());
    for (const auto& core : tt) {
      result.push_back(func(core));
    }
    return ttnte::linalg::TTEngine(result, false);
  }

  // Apply the function in a dense format and then re-decompose
  if (size < max_dense_size) {
    auto dense = tt.to_dense(true);
    dense = func(dense);
    return ttnte::linalg::TTEngine::from_dense(std::move(dense), m_modes,
      n_modes, rounding.eps, rounding.max_rank, true);
  }

  // Use a rank-1 as an initial guess
  auto initial_guess = tt.round(rounding.eps, 1);
  for (auto& core : initial_guess) {
    core.copy_(func(core));
  }

  // Compute with TT-Cross (function_interpolate)
  return ttnte::linalg::function_interpolate(func, {tt}, cross.eps,
    std::move(initial_guess), cross.nswp, cross.kick, cross.max_rank,
    cross.verbose);
}

} // namespace

namespace ttnte::physics::backends {

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::VectorType
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::assemble_ordinates()
{
  if (cache_.has_ordinates()) {
    return cache_.get_ordinates();
  }

  typename Return<Fmt, NumDim>::VectorType ordinates;
  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    if (!angular_qset_->is_tensor_product()) {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_ordinates",
        "The angular quadrature set must be tensor product for tensor "
        "trains");
    }
    auto angular_qset =
      std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);

    // Angular quadrature and total XS
    auto points = angular_qset->get_factored_points();
    torch::Tensor mu = points[0].reshape({1, -1, 1, 1});
    torch::Tensor gamma = points[1].reshape({1, -1, 1, 1});

    ordinates.emplace_back(
      linalg::TTEngine::Tensors {
        torch::sqrt(1 - mu.square()), torch::cos(gamma)},
      false);
    ordinates.emplace_back(
      linalg::TTEngine::Tensors {
        torch::sqrt(1 - mu.square()), torch::sin(gamma)},
      false);
    ordinates.emplace_back(
      linalg::TTEngine::Tensors {mu, torch::ones_like(gamma)}, false);

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    int64_t space_dim = cache_.get_ctrlpts().size();
    if (space_dim > 1) {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_ordinates",
        "Interior loss operator construction for curves only supports slabs");
    }
    if (angular_qset_->is_tensor_product()) {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_ordinates",
        "The angular quadrature set should be 1-D");
    }
    auto angular_qset =
      std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);

    // Angular quadrature
    torch::Tensor mu = angular_qset->get_points().reshape({1, -1, 1, 1});

    ordinates.emplace_back(linalg::TTEngine::Tensors {mu}, false);
  } else {
    throw utils::runtime_error(
      "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_ordinates",
      "This method does not support this format yet");
  }

  // Push to cache
  cache_.set_ordinates(ordinates);

  return ordinates;
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::assemble_basis()
{
  // Check if this has already been calculated
  if (cache_.has_basis()) {
    return cache_.get_basis();
  }

  ReturnType basis;
  if constexpr (Fmt == FormatType::DENSE) {
    basis = block_->evaluate_all_basis(quad_points_, 0, false);

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    // Iterate through each dimension and calculate the B-spline basis
    linalg::TTEngine::Tensors n_cores;
    n_cores.reserve(NumDim);

    for (size_t i = 0; i < block_->get_ndim(); i++) {
      // Basis and points
      const auto& basis = block_->get_basis()[i];
      const auto& points = quad_points_[i];

      // Calculate the basis functions for this specific dimension
      n_cores.push_back(basis.evaluate_all(points));
      n_cores.back().unsqueeze_(0).unsqueeze_(-1);
    }

    // Return if we are only a B-spline
    if (!block_->is_rational()) {
      return linalg::TTEngine(std::move(n_cores), false);
    }

    // Perform Hadamard product with weight TT
    c10::SmallVector<int64_t, 6> n_modes;
    linalg::TTEngine::Tensors d_cores;
    n_modes.reserve(NumDim);
    d_cores.reserve(NumDim);

    const auto& weights = cache_.get_weights();
    for (size_t i = 0; i < NumDim; i++) {
      auto& n_core = n_cores[i];
      const auto& w_core = weights[i];
      n_modes.push_back(n_core.size(2));

      // Hadamard product on this core using broadcasting
      n_core = n_core * w_core;

      // Compute denominator
      d_cores.push_back(n_core.sum(2, true));
    }

    // Create TT for the numerator and denominator
    auto numerator = linalg::TTEngine(std::move(n_cores), false);
    auto denominator = linalg::TTEngine(std::move(d_cores), false);
    denominator.round_(config_->rounding.eps, config_->rounding.max_rank);

    // Compute the reciprocal in TT format
    linalg::TTEngine inv_denominator = apply_mult_separable(
      denominator,
      [](const torch::Tensor& tensor) { return tensor.reciprocal(); },
      config_->rounding, config_->cross, config_->max_dense_size);

    // Expand input dimension of inverse denominator
    inv_denominator.expand_(inv_denominator.get_m_modes(), n_modes);

    // Compute the Hadamard product
    numerator *= inv_denominator;
    numerator.round_(config_->rounding.eps, config_->rounding.max_rank);
    basis = std::move(numerator);

  } else {
    throw utils::runtime_error(
      "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_basis",
      "This method does not support this format yet");
  }

  // Cache result and return
  cache_.set_basis(basis);
  return basis;
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim + 1>::VectorType
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::assemble_basis_ders()
{
  // Check if this has already been calculated
  if (cache_.has_basis() && cache_.has_ders()) {
    if constexpr (Fmt == FormatType::DENSE) {
      return cache_.get_ders();

    } else if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
      c10::SmallVector<ReturnType, NumDim + 1> result = {cache_.get_basis()};
      const auto& ders = cache_.get_ders();
      result.insert(result.end(), ders.begin(), ders.end());
      return result;
    }
  }

  if constexpr (Fmt == FormatType::DENSE) {
    // TODO: There should probably be a better optimized kernel for this
    // Calculate the basis functions and their derivatives
    auto result = block_->evaluate_all_basis(quad_points_, 1, false);

    // Fill cache and return
    cache_.set_basis(result.select(NumDim, 0));
    cache_.set_ders(result);
    return result;

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    c10::SmallVector<linalg::TTEngine, NumDim + 1> result;
    result.reserve(NumDim + 1);

    // Naming convention:
    // der0: zeroth derivative
    // der1: first derivative
    linalg::TTEngine::Tensors der0_cores;
    linalg::TTEngine::Tensors der1_cores;
    der0_cores.reserve(NumDim);
    der1_cores.reserve(NumDim);

    // Evaluate B-spline basis functions and their derivatives
    for (size_t i = 0; i < NumDim; i++) {
      const auto& basis = block_->get_basis()[i];
      const auto& points = quad_points_[i];

      // Calculate the B-spline basis functions for this dimension
      torch::Tensor evals = basis.evaluate_all(points, 1);
      evals.unsqueeze_(0).unsqueeze_(-1);

      // Fill vectors
      der0_cores.push_back(evals.select(2, 0));
      der1_cores.push_back(evals.select(2, 1));
    }

    // Return if the block is only a B-spline
    if (!block_->is_rational()) {
      // Fill return vector:
      // [0]: The evaluated basis functions
      // [1]: Derivative with respect to the first dimension
      // [2]: Derivative with respect to the second dimension
      // ...
      result.emplace_back(std::move(der0_cores), false);

      // Build derivative TTs
      const auto& der0 = result[0];
      for (size_t i = 0; i < NumDim; i++) {
        linalg::TTEngine::Tensors der_cores;

        for (size_t j = 0; j < NumDim; j++) {
          der_cores.push_back(i == j ? der1_cores[j] : der0[j]);
        }

        result.emplace_back(std::move(der_cores), false);
      }

    } else {
      assert(cache_.has_weights());
      const auto& weights = cache_.get_weights();

      // Summed across control points
      linalg::TTEngine::Tensors summed_der0_cores;
      linalg::TTEngine::Tensors summed_der1_cores;
      summed_der0_cores.reserve(NumDim);
      summed_der1_cores.reserve(NumDim);

      // Perform Hadamard products with the weights
      c10::SmallVector<int64_t, 6> m_modes;
      c10::SmallVector<int64_t, 6> n_modes;
      m_modes.reserve(NumDim);
      n_modes.reserve(NumDim);
      for (size_t i = 0; i < NumDim; i++) {
        const auto& w_core = weights[i];
        auto& der0_core = der0_cores[i];
        auto& der1_core = der1_cores[i];

        m_modes.push_back(der0_core.size(1));
        n_modes.push_back(der0_core.size(2));

        der0_core = der0_core * w_core;
        der1_core = der1_core * w_core;

        summed_der0_cores.push_back(der0_core.sum(2, true));
        summed_der1_cores.push_back(der1_core.sum(2, true));
      }

      // Create TTs for the summed cores and round
      linalg::TTEngine summed_der0(std::move(summed_der0_cores), false);
      linalg::TTEngine summed_der1(std::move(summed_der1_cores), false);
      summed_der0.expand_(m_modes, n_modes);
      summed_der1.expand_(m_modes, n_modes);

      // Build numerator of quotient rule
      c10::SmallVector<linalg::TTEngine, NumDim + 1> numerators {
        linalg::TTEngine(std::move(der0_cores), false)};
      numerators.reserve(NumDim + 1);

      const auto& der0 = numerators[0];
      for (size_t i = 0; i < NumDim; i++) {
        linalg::TTEngine::Tensors der_cores;
        linalg::TTEngine::Tensors summed_der_cores;
        der_cores.reserve(NumDim);
        summed_der_cores.reserve(NumDim);

        for (size_t j = 0; j < NumDim; j++) {
          if (i == j) {
            der_cores.push_back(der1_cores[j]);
            summed_der_cores.push_back(summed_der1[j]);
          } else {
            der_cores.push_back(der0[j]);
            summed_der_cores.push_back(summed_der0[j]);
          }
        }

        // Create TTs
        linalg::TTEngine der(std::move(der_cores), false);
        linalg::TTEngine summed_der(std::move(summed_der_cores), false);

        // Compute quotient rule numerator (f'g - fg')
        numerators.push_back(der * summed_der0 - der0 * summed_der);
        numerators.back().round_(
          config_->rounding.eps, config_->rounding.max_rank);
      }

      // Create the denominator by narrowing the summed case
      linalg::TTEngine::Tensors d_cores;
      d_cores.reserve(NumDim);
      for (const auto& core : summed_der0) {
        d_cores.push_back(core.narrow(2, 0, 1));
      }
      linalg::TTEngine denominator(std::move(d_cores), false);
      denominator.round_(config_->rounding.eps, config_->rounding.max_rank);

      // Apply the denominator
      if (denominator.is_rank_one()) {
        // Compute inverse by inverting cores directly
        // {Zeroth derivative
        for (size_t i = 0; i < NumDim; i++) {
          numerators[0][i].div_(denominator[i]);
        }

        // Square the denominator
        for (auto& core : denominator) {
          core.square_();
        }

        // First derivatives
        for (size_t i = 1; i < NumDim + 1; i++) {
          for (size_t j = 0; j < NumDim; j++) {
            numerators[i][j].div_(denominator[j]);
          }
        }
        result = std::move(numerators);

      } else {
        // Compute the inverse of the weight tensor (1 / W)
        auto inv_w = apply_mult_separable(
          denominator,
          [](const torch::Tensor& tensor) { return tensor.reciprocal(); },
          config_->rounding, config_->cross, config_->max_dense_size);

        // Compute the square of the reciprocal
        auto inv_w_sq = inv_w * inv_w;
        inv_w_sq.round_(config_->rounding.eps, config_->rounding.max_rank);

        // Expand input dimension of the reciprocal
        auto inv_denominator = inv_w.expand(m_modes, n_modes);

        // Compute the base case first
        numerators[0] *= inv_denominator;
        numerators[0].round_(config_->rounding.eps, config_->rounding.max_rank);

        // Expand input dimension of the squared reciprocal
        inv_denominator = inv_w_sq.expand(m_modes, n_modes);

        // Apply the inverted denominator to each numerator TT
        for (size_t i = 1; i < NumDim + 1; i++) {
          auto& numerator = numerators[i];
          numerator *= inv_denominator;
          numerator.round_(config_->rounding.eps, config_->rounding.max_rank);
        }
        result = std::move(numerators);
      }
    }
    // Fill cache and return
    cache_.set_basis(result[0]);
    cache_.set_ders(
      c10::SmallVector<ReturnType, NumDim>(result.begin() + 1, result.end()));

    return result;
  }

  throw utils::runtime_error(
    "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_basis_ders",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_scattering_kernel(std::optional<linalg::TTEngine> spatial)
  const
{
  // Constants
  double one = 1.0;
  double two = 2.0;
  double pi = std::numbers::pi;

  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    // Cast to derived class
    if (!angular_qset_->is_tensor_product()) {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_scattering_"
        "kernel",
        "The angular quadrature set must be a tensor product quadrature set");
    }
    auto angular_qset =
      std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);

    // Mode sizes
    const auto& quads = angular_qset->get_quads();
    int64_t nm = quads[0]->get_num_dofs();
    int64_t ng = quads[1]->get_num_dofs();
    const auto& mu = quads[0]->get_points();
    const auto& gamma = quads[1]->get_points();

    // Get tensor options
    const auto& device = mu.device();
    const auto& dtype = mu.scalar_type();
    const auto& options = torch::TensorOptions().device(device).dtype(dtype);
    const auto slice = at::indexing::Slice();

    // Get group-to-group scattering XS tensor for this block
    const auto& scatter_gtg = material_->get_scatter_gtg().to(options);
    int64_t L = scatter_gtg.size(0);

    // Zeroth order, insert the spatial cores if given and energy
    linalg::TTEngine K(Tensors {torch::ones({1, nm, nm, 1}, options),
                         torch::ones({1, ng, ng, 1}, options)},
      false);
    if (spatial.has_value()) {
      K.kron_(*spatial);
    }
    K.kron_(scatter_gtg.narrow(0, 0, 1).unsqueeze(-1));

    // Compute all up to the given order
    double inner_eps = config_->rounding.eps / static_cast<double>(L);
    for (int64_t l = 1; l < L; l++) {
      // The angular quadrature is half-sphere-folded for 2D problems
      // (ProductQuadrature::gauss_legendre_chebyshev's ndim==2 branch,
      // Symmetry::XY_PLANE -- mu in [0,1] only, weights renormalized to
      // sum to 1 under the assumption that psi is symmetric under
      // mu -> -mu, exact for a z-invariant geometry). The true full-sphere
      // scattering integral folds onto this upper hemisphere as
      //   Sum_l (2l+1)/2 * sigma_sl * Sum_j w_j
      //     [P_l(Omega_i.Omega_j) + P_l(Omega_i.Omega_j_bar)] * psi_j
      // where Omega_j_bar is Omega_j's mu -> -mu mirror. Using the
      // associated-Legendre parity identity P_l^m(-mu) = (-1)^(l+m)
      // P_l^m(mu), the m-term of P_l(Omega_i.Omega_j) +
      // P_l(Omega_i.Omega_j_bar) doubles when m and l share parity and
      // cancels exactly otherwise -- that factor of 2 exactly cancels the
      // leading 1/2 above, so the net effect is: sum only m = l%2,
      // l%2+2, ..., l (same parity as l), dropping every opposite-parity m
      // entirely, with no other change to each surviving term's own
      // normalization. (A full-sphere quadrature, e.g. NumDim==1's polar-
      // only case elsewhere in this file, would need every m instead --
      // this parity restriction is specific to the folded convention
      // here.)
      const double l_f = l;
      const int64_t m_start = l % 2;
      const int64_t num_m = (l - m_start) / 2 + 1; // surviving m count
      const int32_t num_terms = static_cast<int32_t>(
        m_start == 0 ? 2 * num_m - 1 : 2 * num_m); // m=0 gets only cos

      linalg::TTEngine Yl(linalg::TTEngine::Tensors {
        torch::zeros({1, nm, nm, num_terms}, options),
        torch::zeros({num_terms, ng, ng, 1}, options)});

      int64_t col = 0;
      for (int64_t m = m_start; m <= l; m += 2) {
        const double m_f = m;

        // Compute the normalization factor. The addition theorem's m != 0
        // term carries a factor of 2*(l-m)!/(l+m)! split across the cos and
        // sin columns below (P_l(cos_theta) = P_l^0(mu)P_l^0(mu') +
        // 2*sum_{m=1}^l [(l-m)!/(l+m)!] P_l^m(mu)P_l^m(mu') cos(m*(gamma -
        // gamma'))): squaring this per-column scale in the outer product
        // below must recover that 2x once, not once per column, so the
        // shared per-column prefactor is sqrt(2), not 2.
        double log_front = std::log(two * l_f + one);
        double log_factor =
          std::lgamma(l_f - m_f + one) - std::lgamma(l_f + m_f + one);
        double scale =
          (m != 0 ? std::sqrt(two) : one) *
          std::exp(static_cast<double>(0.5) * (log_front + log_factor));

        // Compute associated Legendre polynomials and the normalization
        // factor
        auto plm = scale * math::special::assoc_legendre(l, m, mu, false);
        auto plm_outer = torch::outer(plm, plm);

        // Cosine column (always present for a surviving m)
        Yl[0].index_put_({0, slice, slice, col}, plm_outer);
        auto cos_gamma = torch::cos(m_f * gamma);
        Yl[1].index_put_(
          {col, slice, slice, 0}, torch::outer(cos_gamma, cos_gamma));
        col++;

        if (m != 0) {
          // Sine column -- m=0 has sin(0*gamma) = 0 identically, no column
          // needed (matches num_terms' m_start==0 branch using 2*num_m-1).
          Yl[0].index_put_({0, slice, slice, col}, plm_outer);
          auto sin_gamma = torch::sin(m_f * gamma);
          Yl[1].index_put_(
            {col, slice, slice, 0}, torch::outer(sin_gamma, sin_gamma));
          col++;
        }
      }
      assert(col == num_terms);

      // Round the result down
      Yl.round_(config_->rounding.eps, config_->rounding.max_rank);

      // Append spatial and energy cores
      if (spatial.has_value()) {
        Yl.kron_(*spatial);
      }
      Yl.kron_(scatter_gtg.narrow(0, l, 1).unsqueeze(-1));

      // Add this moment to the scattering kernel
      K += Yl;
      K.round_(inner_eps, config_->rounding.max_rank);
    }

    return K;

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    assert(std::dynamic_pointer_cast<math::QuadratureSet1D>(angular_qset_));

    // Cast to derived class
    auto angular_qset =
      std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);

    // Mode sizes (Only polar angle mu exists)
    int64_t nm = angular_qset->get_num_dofs();
    const auto& mu = angular_qset->get_points();

    // Get tensor options
    const auto& device = mu.device();
    const auto& dtype = mu.scalar_type();
    const auto options_f = torch::TensorOptions().device(device).dtype(dtype);
    const auto slice = at::indexing::Slice();

    // Get group-to-group scattering XS tensor for this block
    const auto& scatter_gtg = material_->get_scatter_gtg().to(options_f);
    int64_t L = scatter_gtg.size(0);

    // Initialize the base case 0th order un-normalized Legendre polynomial
    linalg::TTEngine K(Tensors {torch::ones({1, nm, nm, 1}, options_f)}, false);
    if (spatial.has_value()) {
      K.kron_(*spatial);
    }
    K.kron_(scatter_gtg.narrow(0, 0, 1).unsqueeze(-1));

    double inner_eps = config_->rounding.eps / static_cast<double>(L);
    for (int64_t l = 1; l < L; l++) {
      double l_f = static_cast<double>(l);

      linalg::TTEngine Pl(
        linalg::TTEngine::Tensors {torch::zeros({1, nm, nm, 1}, options_f)});

      // Compute the lth un-normalized Legendre polynomial and the scale
      double scale = std::sqrt(two * l_f + one);
      auto pl = scale * math::special::legendre(l, mu, false);

      // Insert the outer product into the angle core
      Pl[0].index_put_({0, slice, slice, 0}, torch::outer(pl, pl));

      // Insert spatial and energy core
      if (spatial.has_value()) {
        Pl.kron_(*spatial);
      }
      Pl.kron_(scatter_gtg.narrow(0, l, 1).unsqueeze(-1));

      // Add this moment to the scattering kernel
      K += Pl;
    }
    K.round_(inner_eps, config_->rounding.max_rank);

    return K;
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_scattering_kernel",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_angular_integral() const
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    // Cast to derived class
    if (!angular_qset_->is_tensor_product()) {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_scattering_"
        "kernel",
        "The angular quadrature set must be a tensor product quadrature set");
    }
    auto angular_qset =
      std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);

    // Quadratures
    const auto& quads = angular_qset->get_quads();
    assert(quads.size() == 2);
    int64_t nm = quads[0]->get_num_dofs();
    int64_t ng = quads[1]->get_num_dofs();
    c10::SmallVector<int64_t, 6> modes = {nm, ng};

    // Angular integration operator
    return linalg::TTEngine(
      linalg::TTEngine::Tensors {quads[0]->get_weights().reshape({1, 1, -1, 1}),
        quads[1]->get_weights().reshape({1, 1, -1, 1})})
      .expand(modes, modes);

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    assert(std::dynamic_pointer_cast<math::QuadratureSet1D>(angular_qset_));

    // Cast to derived class
    auto angular_qset =
      std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);

    // Quadratures
    int64_t nm = angular_qset->get_num_dofs();

    // Angular integration operator
    return linalg::TTEngine(
      linalg::TTEngine::Tensors {
        angular_qset->get_weights().reshape({1, 1, -1, 1})})
      .expand({nm}, {nm});
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_angular_integral",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::MatrixType
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::assemble_jacobian()
{
  if (cache_.has_jacobian()) {
    return cache_.get_jacobian();
  }

  ReturnMatrixType jacobian;
  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    // Get the basis function derivatives and weights
    if (!cache_.has_ders()) {
      assemble_basis_ders();
    }
    const auto& ders = cache_.get_ders();
    const auto& ctrlpts = cache_.get_ctrlpts();

    int64_t space_dim = ctrlpts.size();
    if (space_dim > 3) {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::"
        "assemble_jacobian",
        "The number of spatial dimensions maxes out at 3");
    }

    // Calculate the Jacobian
    for (int64_t i = 0; i < space_dim; i++) {
      for (int64_t j = 0; j < NumDim; j++) {
        jacobian[i][j] = linalg::mm(ders[j], ctrlpts[i].transpose());
      }
    }

  } else {
    throw utils::runtime_error(
      "ttnte::physics::DIGAFirstOrderTransportBackend::"
      "assemble_jacobian",
      "This method does not support this format yet");
  }

  // Fill cache and return
  cache_.set_jacobian(jacobian);
  return jacobian;
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_integral_mapping()
{
  if (cache_.has_mapping()) {
    return cache_.get_mapping();
  }

  ReturnType mapping;
  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    if (!cache_.has_jacobian()) {
      assemble_jacobian();
    }
    const auto& J = cache_.get_jacobian();
    int64_t space_dim = cache_.get_ctrlpts().size();

    // Calculate the determinant (area mapping)
    if constexpr (NumDim == 1) {
      if (space_dim == 1) {
        mapping = J[0][0];

      } else {
        linalg::TTEngine squared_sum = J[0][0] * J[0][0];
        for (int64_t i = 1; i < space_dim; i++) {
          squared_sum += J[i][0] * J[i][0];
        }

        // Compute the square root
        mapping = apply_mult_separable(
          std::move(squared_sum),
          [](const torch::Tensor& tensor) { return tensor.sqrt(); },
          config_->rounding, config_->cross, config_->max_dense_size);
      }

    } else if constexpr (NumDim == 2) {
      if (space_dim == 1) {
        throw utils::runtime_error(
          "ttnte::physics::DIGAFirstOrderTransportBackend::"
          "assemble_integral_mapping",
          "The patch must be 2 or 3-D for a 2-D parametric surface");

      } else if (space_dim == 2) {
        // Standard flat 2-D plane determinant
        mapping = J[0][0] * J[1][1] - J[0][1] * J[1][0];

      } else {
        // 2-D parametric surface in 3-D space, use the magnitude of cross
        // product
        auto cx = J[1][0] * J[2][1] - J[2][0] * J[1][1];
        auto cy = J[2][0] * J[0][1] - J[0][0] * J[2][1];
        auto cz = J[0][0] * J[1][1] - J[1][0] * J[0][1];

        double strict_eps = config_->rounding.eps / 3.0;
        cx.round_(strict_eps, config_->rounding.max_rank);
        cy.round_(strict_eps, config_->rounding.max_rank);
        cz.round_(strict_eps, config_->rounding.max_rank);

        auto squared_sum = (cx * cx) + (cy * cy) + (cz * cz);
        squared_sum.round_(config_->rounding.eps, config_->rounding.max_rank);

        mapping = apply_mult_separable(
          std::move(squared_sum),
          [](const torch::Tensor& tensor) { return tensor.sqrt(); },
          config_->rounding, config_->cross, config_->max_dense_size);
      }

    } else if constexpr (NumDim == 3) {
      if (space_dim == 3) {
        auto term1 = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]);
        auto term2 = J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]);
        auto term3 = J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
        mapping = term1 - term2 + term3;

      } else {
        throw utils::runtime_error(
          "ttnte::physics::DIGAFirstOrderTransportBackend::"
          "assemble_integral_mapping",
          "The patch must be 3-D for a 3-D parametric surface");
      }
    } else {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::"
        "assemble_integral_mapping",
        "Parametric dimensions higher than 3-D are not supported");
    }

    // Round the result
    mapping.round_(config_->rounding.eps, config_->rounding.max_rank);

    // Compute the parent mapping determinant
    const auto& options = torch::TensorOptions()
                            .device(mapping.get_device())
                            .dtype(mapping.get_dtype());
    for (int64_t i = 0; i < NumDim; i++) {
      const auto& b = block_->get_basis()[i];
      const auto& quad = spatial_qset_->get_quads()[i];

      // Get the unique knots in the knot vector
      torch::Tensor unique_knots = b.get_unique_knots().to(options);

      // Compute derivatives of affine mapping
      // Note not dividing by 2 because that cancels out with the weight of the
      // quadrature
      torch::Tensor jac_1d =
        (unique_knots.diff().unsqueeze(1) * quad->get_weights().unsqueeze(0))
          .flatten()
          .reshape({1, -1, 1, 1});

      // Apply rank-one Hadamard product
      mapping[i].mul_(jac_1d);
    }

  } else {
    throw utils::runtime_error(
      "ttnte::physics::DIGAFirstOrderTransportBackend::"
      "assemble_integral_mapping",
      "This method does not support this format yet");
  }

  // Adjust for right handed versus left handed coordinate system
  mapping *= block_->get_orientation();

  // Fill cache and return
  cache_.set_mapping(mapping);
  return mapping;
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::MatrixType DGFirstOrderTransportBackend<
  cad::Patch, Fmt, NumDim>::assemble_jacobian_inverse()
{
  if (cache_.has_jacobian_inverse()) {
    return cache_.get_jacobian_inverse();
  }

  ReturnMatrixType jacobian_inverse;
  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim <= 3) {
    // Get the Jacobian
    if (!cache_.has_jacobian()) {
      assemble_jacobian();
    }
    int64_t space_dim = cache_.get_ctrlpts().size();
    const auto& jacobian = cache_.get_jacobian();
    const auto& device = jacobian[0][0].get_device();
    const auto& dtype = jacobian[0][0].get_dtype();

    // Dimension checks
    if (space_dim > 3) {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::"
        "assemble_jacobian_inverse",
        "Physical dimensions higher than 3-D are not supported");
    } else if (space_dim < NumDim) {
      throw utils::runtime_error(
        "ttnte::physics::DIGAFirstOrderTransportBackend::"
        "assemble_jacobian_inverse",
        "There must be at least as many physical dimensions as parametric");
    }

    if constexpr (NumDim == 1) {
      if (space_dim == 1) {
        jacobian_inverse[0][0] =
          linalg::TTEngine({jacobian[0][0][0].reciprocal()}, false);
      } else {
        // Calculate mapping
        torch::Tensor g = jacobian[0][0][0].square();
        for (int64_t i = 1; i < space_dim; i++) {
          g.add_(jacobian[i][0][0].square());
        }

        // Divide each by the mapping
        for (int64_t i = 0; i < space_dim; i++) {
          jacobian_inverse[i][0] =
            linalg::TTEngine({jacobian[i][0][0] / g}, false);
        }
      }

    } else {
      if (!config_->cross_jacobian_inverse) {
        if constexpr (NumDim == 2) {
          if (space_dim == 2) {
            // Compute the determinant of a 2x2
            linalg::TTEngine det =
              jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
            det.round_(config_->rounding.eps, config_->rounding.max_rank);

            // Compute the reciprocal
            auto inv_det = apply_mult_separable(
              det,
              [](const torch::Tensor& tensor) { return tensor.reciprocal(); },
              config_->rounding, config_->cross, config_->max_dense_size);

            // Finish the inverse in 2-D
            for (int64_t i = 0; i < NumDim; i++) {
              for (int64_t j = 0; j < NumDim; j++) {
                int64_t i_inv = 1 - i;
                int64_t j_inv = 1 - j;
                double scale = i == j ? 1.0 : -1.0;

                jacobian_inverse[i][j] =
                  scale * inv_det * jacobian[i_inv][j_inv];
                jacobian_inverse[i][j].round_(
                  config_->rounding.eps, config_->rounding.max_rank);
              }
            }

          } else {
            double inner_eps = config_->rounding.eps / 3.0;

            // Compute by the pseudo-inverse in TT format for 3x2 Jacobian
            auto g00 = (jacobian[0][0] * jacobian[0][0] +
                        jacobian[1][0] * jacobian[1][0] +
                        jacobian[2][0] * jacobian[2][0])
                         .round(inner_eps, config_->rounding.max_rank);
            auto g01 = (jacobian[0][0] * jacobian[0][1] +
                        jacobian[1][0] * jacobian[1][1] +
                        jacobian[2][0] * jacobian[2][1])
                         .round(inner_eps, config_->rounding.max_rank);
            auto g11 = (jacobian[0][1] * jacobian[0][1] +
                        jacobian[1][1] * jacobian[1][1] +
                        jacobian[2][1] * jacobian[2][1])
                         .round(inner_eps, config_->rounding.max_rank);

            auto det_g = g00 * g11 - g01 * g01;
            det_g.round_(config_->rounding.eps, config_->rounding.max_rank);

            // Compute the reciprocal of the determinant
            auto inv_det_g = apply_mult_separable(
              det_g,
              [](const torch::Tensor& tensor) { return tensor.reciprocal(); },
              config_->rounding, config_->cross, config_->max_dense_size);

            // Build the pseudo-inverse
            for (int64_t i = 0; i < space_dim; i++) {
              for (int64_t j = 0; j < NumDim; j++) {
                jacobian_inverse[i][j] =
                  (j == 0 ? (jacobian[i][0] * g11 - jacobian[i][1] * g01)
                          : (jacobian[i][1] * g00 - jacobian[i][0] * g01)) *
                  inv_det_g;
                jacobian_inverse[i][j].round_(
                  config_->rounding.eps, config_->rounding.max_rank);
              }
            }
          }

        } else {
          double inner_eps = config_->rounding.eps / 3.0;

          // 3-D volume case (3x3 matrix)
          auto det_j =
            jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] -
                               jacobian[1][2] * jacobian[2][1])
                               .round(inner_eps, config_->rounding.max_rank) -
            jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] -
                               jacobian[1][2] * jacobian[2][0])
                               .round(inner_eps, config_->rounding.max_rank) +
            jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] -
                               jacobian[1][1] * jacobian[2][0])
                               .round(inner_eps, config_->rounding.max_rank);
          det_j.round_(config_->rounding.eps, config_->rounding.max_rank);

          // Compute the reciprocal of the determinant
          auto inv_det_j = apply_mult_separable(
            det_j,
            [](const torch::Tensor& tensor) { return tensor.reciprocal(); },
            config_->rounding, config_->cross, config_->max_dense_size);

          // Build cofactors algebraically
          for (int64_t i = 0; i < 3; i++) {
            for (int64_t j = 0; j < 3; j++) {
              int64_t r0 = (i == 0) ? 1 : 0;
              int64_t r1 = (i == 2) ? 1 : 2;
              int64_t c0 = (j == 0) ? 1 : 0;
              int64_t c1 = (j == 2) ? 1 : 2;

              jacobian_inverse[i][j] =
                inv_det_j * (jacobian[r0][c0] * jacobian[r1][c1] -
                              jacobian[r0][c1] * jacobian[r1][c0]);
              jacobian_inverse[i][j].round_(
                config_->rounding.eps, config_->rounding.max_rank);

              if ((i + j) % 2 != 0) {
                jacobian_inverse[i][j].neg_();
              }
            }
          }
        }

      } else {
        // Get mode sizes for TT-cross
        std::vector<int64_t> N;
        N.reserve(NumDim);
        for (const auto& core : jacobian[0][0]) {
          N.push_back(core.size(1));
        }

        // Iterate through each component of the Jacobian and apply TT-cross
        for (int64_t i = 0; i < space_dim; i++) {
          for (int64_t j = 0; j < NumDim; j++) {
            // TT-cross function
            auto func =
              [&](const torch::Tensor& spatial_indices) -> torch::Tensor {
              if constexpr (NumDim == 2) {
                // Compute the components of the Jacobian
                std::array<std::array<torch::Tensor, 2>, 3> jac;
                jac[0][0] = jacobian[0][0].evaluate_at(spatial_indices);
                jac[0][1] = jacobian[0][1].evaluate_at(spatial_indices);
                jac[1][0] = jacobian[1][0].evaluate_at(spatial_indices);
                jac[1][1] = jacobian[1][1].evaluate_at(spatial_indices);

                if (space_dim == 2) {
                  // Compute the inverse of a 2x2 matrix
                  int64_t i_inv = 1 - i;
                  int64_t j_inv = 1 - j;
                  double scale = i == j ? 1.0 : -1.0;
                  return scale /
                         (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]) *
                         jac[i_inv][j_inv];

                } else {
                  // Compute z-components
                  jac[2][0] = jacobian[2][0].evaluate_at(spatial_indices);
                  jac[2][1] = jacobian[2][1].evaluate_at(spatial_indices);

                  // Compute the square components of the pseudo-inverse
                  auto g00 = jac[0][0].square() + jac[1][0].square() +
                             jac[2][0].square();
                  auto g01 = jac[0][0] * jac[0][1] + jac[1][0] * jac[1][1] +
                             jac[2][0] * jac[2][1];
                  auto g11 = jac[0][1].square() + jac[1][1].square() +
                             jac[2][1].square();

                  // Compute the component of the inverse of a 3x2 matrix using
                  // pseudo-inverse
                  return (j == 0 ? (jac[i][0] * g11 - jac[i][1] * g01)
                                 : (jac[i][1] * g00 - jac[i][0] * g01)) /
                         (g00 * g11 - g01.square());
                }
              } else {
                // 3-D volume case (3x3 matrix)
                std::array<std::array<torch::Tensor, 3>, 3> jac;
                for (int r = 0; r < 3; r++) {
                  for (int c = 0; c < 3; c++) {
                    jac[r][c] = jacobian[r][c].evaluate_at(spatial_indices);
                  }
                }

                torch::Tensor det_j =
                  jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1]) -
                  jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0]) +
                  jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);

                // Logic for the transposed inverse
                int64_t r0 = (i == 0) ? 1 : 0;
                int64_t r1 = (i == 2) ? 1 : 2;
                int64_t c0 = (j == 0) ? 1 : 0;
                int64_t c1 = (j == 2) ? 1 : 2;

                torch::Tensor cofactor =
                  jac[r0][c0] * jac[r1][c1] - jac[r0][c1] * jac[r1][c0];

                if ((i + j) % 2 != 0) {
                  cofactor = -cofactor;
                }

                return cofactor / det_j;
              }
            };

            // Run TT-cross on all elements
            jacobian_inverse[i][j] =
              linalg::dmrg_cross(func, N, config_->cross.eps,
                config_->cross.nswp, std::nullopt, config_->cross.kick,
                config_->cross.max_rank, config_->cross.verbose, device, dtype);
          }
        }
      }
    }

  } else {
    throw utils::runtime_error(
      "ttnte::physics::DIGAFirstOrderTransportBackend::"
      "assemble_jacobian_inverse",
      "This method does not support this format yet");
  }

  // Cache result and return
  cache_.set_jacobian_inverse(jacobian_inverse);
  return jacobian_inverse;
}

template<FormatType Fmt, int64_t NumDim>
linalg::Operator
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::assemble_loss_operator()
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    // Assemble components
    auto basis = assemble_basis_ders();
    auto jac_inv = assemble_jacobian_inverse();

    // Sizes
    int64_t space_dim = cache_.get_ctrlpts().size();
    auto m_modes = basis[0].get_m_modes();
    auto n_modes = basis[0].get_n_modes();
    auto options = torch::TensorOptions()
                     .device(basis[0].get_device())
                     .dtype(basis[0].get_dtype());

    // Inner loop truncation tolerance
    double inner_eps = config_->rounding.eps / static_cast<double>(NumDim + 1);

    // Apply integral mapping with basis evaluation
    linalg::TTEngine mapped_basis;
    if (cache_.has_mapped_basis()) {
      mapped_basis = cache_.get_mapped_basis();
    } else {
      auto mapping = assemble_integral_mapping();

      mapped_basis = basis[0] * mapping.expand(m_modes, n_modes);
      mapped_basis.round_(inner_eps, config_->rounding.max_rank);

      // Add it to the cache
      cache_.set_mapped_basis(mapped_basis);
    }

    // Total XS
    torch::Tensor total =
      material_->get_total().to(options).reshape({1, -1, 1, 1});

    // Assemble the ordinates in TT format
    auto ordinates = assemble_ordinates();

    if (NumDim > 1) {
      // Total interaction term
      linalg::TTEngine H(
        linalg::TTEngine::Tensors {
          torch::ones_like(ordinates[0][0]), torch::ones_like(ordinates[0][1])},
        false);
      H.kron_(linalg::mm(basis[0].transpose(), mapped_basis)
          .round(inner_eps, config_->rounding.max_rank));
      H.kron_(total);

      // Compute gradient dotted with the neutron direction
      for (int64_t i = 0; i < space_dim; i++) {
        auto grad_i = jac_inv[i][0].expand(m_modes, n_modes) * basis[1];
        for (int64_t j = 1; j < NumDim; j++) {
          grad_i += jac_inv[i][j].expand(m_modes, n_modes) * basis[j + 1];
        }
        grad_i.round_(inner_eps, config_->rounding.max_rank);

        auto H_i =
          ordinates[i].kron(linalg::mm(grad_i.transpose(), mapped_basis));
        H_i.kron_(torch::ones_like(total));

        H -= H_i;
      }
      H.round_(config_->rounding.eps, config_->rounding.max_rank);

      // Diagonalize the angular and energy cores;
      H.diagonalize_({0, 1, H.size() - 1});

      return linalg::Operator(std::move(H));

    } else if constexpr (NumDim == 1) {
      // Total interaction term
      linalg::TTEngine H(
        linalg::TTEngine::Tensors {torch::ones_like(ordinates[0][0])}, false);
      H.kron_(linalg::mm(basis[0].transpose(), mapped_basis)
          .round(inner_eps, config_->rounding.max_rank));
      H.kron_(total);

      // Compute the gradient
      auto grad = jac_inv[0][0].expand(m_modes, n_modes) * basis[1];

      H -= ordinates[0]
             .kron(linalg::mm(grad.transpose(), mapped_basis))
             .kron(torch::ones_like(total));
      H.round_(config_->rounding.eps, config_->rounding.max_rank);

      // Diagonalize angle and energy
      H.diagonalize_({0, H.size() - 1});

      return linalg::Operator(std::move(H));
    }
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_loss_operator",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
linalg::Operator DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_scatter_operator()
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    // Get the components for the scattering operator
    auto basis = assemble_basis();

    // Apply integral mapping with basis evaluation
    linalg::TTEngine mapped_basis;
    if (cache_.has_mapped_basis()) {
      mapped_basis = cache_.get_mapped_basis();
    } else {
      auto mapping = assemble_integral_mapping();

      mapped_basis =
        basis * mapping.expand(basis.get_m_modes(), basis.get_n_modes());
      mapped_basis.round_(
        config_->rounding.eps / static_cast<double>(NumDim + 1),
        config_->rounding.max_rank);

      // Add it to the cache
      cache_.set_mapped_basis(mapped_basis);
    }

    // Compute the integrated volume
    auto S = linalg::mm(basis.transpose(), std::move(mapped_basis));
    S.round_(config_->rounding.eps, config_->rounding.max_rank);

    // Compute the scattering kernel with the spatial cores in place
    S = assemble_scattering_kernel(std::move(S));

    // Apply the angular integral
    if constexpr (NumDim > 1) {
      S = apply_angular_weights(S, {0, 1});
    } else {
      S = apply_angular_weights(S, {0});
    }

    return linalg::Operator(std::move(S));
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_scatter_operator",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
linalg::Operator DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_fission_operator()
{
  if (!material_->is_fissile()) {
    return linalg::Operator();
  }

  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    int64_t space_dim = cache_.get_ctrlpts().size();

    // Get the components for the scattering operator
    auto basis = assemble_basis();

    // Apply integral mapping with basis evaluation
    linalg::TTEngine mapped_basis;
    if (cache_.has_mapped_basis()) {
      mapped_basis = cache_.get_mapped_basis();
    } else {
      auto mapping = assemble_integral_mapping();

      mapped_basis =
        basis * mapping.expand(basis.get_m_modes(), basis.get_n_modes());
      mapped_basis.round_(
        config_->rounding.eps / static_cast<double>(NumDim + 1),
        config_->rounding.max_rank);

      // Add it to the cache
      cache_.set_mapped_basis(mapped_basis);
    }

    // Get angular components
    linalg::TTEngine F = assemble_angular_integral();

    // Compute the integrated volume component
    F.kron_(linalg::mm(basis.transpose(), std::move(mapped_basis))
        .round(config_->rounding.eps, config_->rounding.max_rank));

    // Add an energy core
    int64_t num_groups = material_->get_num_groups();
    F.kron_(torch::outer(material_->get_chi(), material_->get_nu_fission())
        .reshape({1, num_groups, num_groups, 1}));

    return linalg::Operator(std::move(F));
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_scatter_operator",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
linalg::State
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::assemble_source(
  const FixedSource& source)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    auto options = torch::TensorOptions()
                     .device(block_->get_device())
                     .dtype(block_->get_dtype());

    // Angular "ones" cores (uniform-across-ordinates, for the isotropic
    // branch, and the angle-doesn't-matter placeholder for the function
    // branch's non-angular coordinate TTEngines) and the real angular VALUE
    // cores (the function branch's angular coordinate TTEngines), sized per
    // NumDim exactly like assemble_angular_integral()/assemble_ordinates().
    linalg::TTEngine::Tensors angular_ones_cores;
    linalg::TTEngine::Tensors angular_value_cores;
    if constexpr (NumDim > 1) {
      if (!angular_qset_->is_tensor_product()) {
        throw utils::runtime_error(
          "ttnte::physics::DGFirstOrderTransportBackend::assemble_source",
          "The angular quadrature set must be a tensor product quadrature set");
      }
      auto angular_qset =
        std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);
      const auto& quads = angular_qset->get_quads();
      angular_ones_cores.push_back(
        torch::ones({1, quads[0]->get_num_dofs(), 1, 1}, options));
      angular_ones_cores.push_back(
        torch::ones({1, quads[1]->get_num_dofs(), 1, 1}, options));
      angular_value_cores.push_back(
        quads[0]->get_points().reshape({1, -1, 1, 1}));
      angular_value_cores.push_back(
        quads[1]->get_points().reshape({1, -1, 1, 1}));
    } else {
      auto angular_qset =
        std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);
      angular_ones_cores.push_back(
        torch::ones({1, angular_qset->get_num_dofs(), 1, 1}, options));
      angular_value_cores.push_back(
        angular_qset->get_points().reshape({1, -1, 1, 1}));
    }

    int64_t num_groups = material_->get_num_groups();
    std::vector<linalg::TTEngine> terms;

    // ---- Isotropic branch ----
    //
    // The load-vector entry for basis function R_i is
    //   integral_over_Vhat( R_i(xhat) * Q(xhat) * |J(xhat)| ) dVhat.
    // Here Q is `isotropic_strength`: a single physical-unit value per
    // energy group, i.e. Q(xhat) == Q_g for every xhat (NOT a NURBS field
    // with its own per-control-point coefficients Q_ijk the way the state
    // psi or a mass-matrix operator's trial function would be). Because Q
    // is constant in xhat, it factors outside the integral instead of
    // needing evaluation at every quadrature point:
    //   Q_g * integral_over_Vhat( R_i(xhat) * |J(xhat)| ) dVhat.
    // `mapping` (assemble_integral_mapping()) already IS the quadrature
    // approximation of that remaining integral's non-basis part, |J(xhat_q)|
    // * w_q at each quad point; `basis` (assemble_basis()) already IS
    // R_i(xhat_q) at those same points (both are evaluated at quadrature
    // points internally, same as every other operator in this file). So
    // `mm(basis.transpose(), mapping)` performs exactly the quadrature sum
    // sum_q( R_i(xhat_q) * |J(xhat_q)| * w_q ) == integral_over_Vhat(
    // R_i(xhat) * |J(xhat)| ) dVhat, and the isotropic_strength factor Q_g is
    // then kron_'d in afterward -- mathematically identical to multiplying Q
    // into the integrand before integrating, since Q doesn't vary with xhat.
    // This is why there's no second per-quadrature-point evaluation of Q
    // here: for a spatially-constant source there's nothing to evaluate.
    if (source.isotropic_strength.has_value()) {
      TORCH_CHECK(source.isotropic_strength->numel() == num_groups,
        "FixedSource::isotropic_strength must have one entry per energy "
        "group");

      linalg::TTEngine isotropic(angular_ones_cores, false);

      // Spatial load vector: mv(basis^T, mapping) -- NOT the mass-matrix-
      // style mapped_basis (basis * mapping) used by
      // assemble_fission_operator()/assemble_scatter_operator(); the raw
      // (n=1) mapping already gives the correct (m=ctrlpts, n=1) shape for
      // integrating a constant density against the DG test basis.
      auto basis = assemble_basis();
      auto mapping = assemble_integral_mapping();
      linalg::TTEngine spatial = linalg::mm(basis.transpose(), mapping);
      spatial.round_(config_->rounding.eps / static_cast<double>(NumDim + 1),
        config_->rounding.max_rank);
      isotropic.kron_(spatial);

      // No angular normalization needed: angular quadrature weights are
      // always normalized to sum to 1 (see QuadratureSet1D::gauss_legendre/
      // gauss_chebyshev, `weights / weights.sum()`), not to
      // get_weighting_factor() -- so a uniform per-direction value of Q
      // already integrates (via qset.integrate(), which contracts against
      // those sum-to-1 weights) to exactly Q, matching the physical-units
      // convention directly.
      isotropic.kron_(
        source.isotropic_strength->reshape({1, num_groups, 1, 1}).to(options));

      terms.emplace_back(std::move(isotropic));
    }

    // ---- Pre-projected isotropic load branch ----
    //
    // Here the caller has ALREADY performed the Galerkin projection of a
    // (possibly spatially varying) angle-independent density q_g(x) onto the
    // DG test basis, b_{i,g} = integral( q_g(x) * R_i(x) ) dV, and handed it
    // over as a dense (space x NumDim, energy) tensor. Since it is already
    // the load vector (same units as the isotropic branch's
    // `mm(basis^T, mapping) x Q_g`), all that remains is to compress it to a
    // TT and broadcast it uniformly across ordinates -- again with no angular
    // normalization, because the angular weights sum to 1 (see the isotropic
    // branch).
    if (source.projected_isotropic_source.has_value()) {
      const auto& dense = *source.projected_isotropic_source;
      const std::string ctx =
        "ttnte::physics::DGFirstOrderTransportBackend::assemble_source";

      if (!dense.defined() || dense.dim() != NumDim + 1) {
        throw utils::runtime_error(
          ctx, "FixedSource::projected_isotropic_source must be a dense tensor "
               "with NumDim + 1 = " +
                 std::to_string(NumDim + 1) +
                 " dimensions (control points per parametric dimension, then "
                 "energy groups)");
      }
      for (int64_t k = 0; k <= NumDim; k++) {
        int64_t expected =
          k < NumDim ? block_->get_ctrlpts_size(k) : num_groups;
        if (dense.size(k) != expected) {
          throw utils::runtime_error(ctx,
            "FixedSource::projected_isotropic_source dimension " +
              std::to_string(k) + " has size " + std::to_string(dense.size(k)) +
              " but the expected " +
              (k < NumDim
                  ? "number of control points along parametric dimension " +
                      std::to_string(k)
                  : std::string("number of energy groups")) +
              " is " + std::to_string(expected));
        }
      }

      linalg::TTEngine load(angular_ones_cores, false);
      load.kron_(linalg::TTEngine::from_dense(
        dense.to(options), config_->rounding.eps, config_->rounding.max_rank));
      terms.emplace_back(std::move(load));
    }

    // ---- Arbitrary function (MMS or otherwise) branch ----
    //
    // Here Q(xhat) is a genuine user-supplied function of physical
    // coordinates (and, unlike the isotropic branch, may also vary over
    // angle/energy) -- it can no longer be factored outside the integral, so
    // it DOES need evaluating at every quadrature point:
    //   integral( R_i(xhat) * Q(x, Omega, g) * |J(xhat)| ) dVhat.
    // Since NumDim-dimensional TT-cross sampling needs the joint sample
    // points, not a separable per-axis quadrature rule, `func` is sampled at
    // the (angle, space x NumDim, energy) grid via `function_interpolate()`
    // (TT-cross), producing `function_source` == Q(x_q, Omega_q, g) as a TT
    // over that same combined axis structure `assemble_ordinates()`/
    // `assemble_angular_integral()` use. `mapping_full` broadcasts the same
    // |J(xhat_q)| * w_q factor used above across angle/energy, and the
    // Hadamard product `function_source *= mapping_full` gives
    // Q(x_q, Omega_q, g) * |J(xhat_q)| * w_q at each sample -- the full
    // quadrature-weighted integrand, still expressed pointwise (not yet
    // projected onto the DG test basis). `full_basis` sandwiches the actual
    // spatial `basis` (R_i(xhat_q)) between angle/energy IDENTITY operators
    // so a single `mm(full_basis, function_source)` performs the remaining
    // sum_q( R_i(xhat_q) * (...) ) contraction -- the Galerkin projection --
    // exactly mirroring the isotropic branch's `mm(basis.transpose(),
    // mapping)`, just against a function-sampled integrand instead of a
    // constant one.
    if (source.function.has_value()) {
      const auto& func = *source.function;

      // The assembler's own quad_points_ are parametric -- map through the
      // patch's own geometry evaluation so the user's function is always
      // called with PHYSICAL coordinates, not parametric ones.
      torch::Tensor phys = block_->evaluate(quad_points_);
      int64_t phys_dim = phys.size(-1);

      linalg::TTEngine::Tensors spatial_ones_cores;
      spatial_ones_cores.reserve(NumDim);
      for (int64_t d = 0; d < NumDim; d++) {
        spatial_ones_cores.push_back(
          torch::ones({1, quad_points_[d].size(0), 1, 1}, options));
      }
      torch::Tensor energy_ones = torch::ones({1, num_groups, 1, 1}, options);
      torch::Tensor energy_values =
        torch::arange(num_groups, options).reshape({1, -1, 1, 1});

      // Coordinate TTEngines, all sharing the same combined [angle, space
      // (x NumDim), energy] core structure -- required so TT-cross can
      // jointly sample every axis at the same multi-index (mirrors
      // TTEngine::meshgrid()'s "each axis real in its own core, ones
      // elsewhere" convention, generalized to the spatial coordinates,
      // which -- for a curved patch -- aren't separable per parametric
      // dimension the way the angular/energy axes are).
      std::vector<linalg::TTEngine> xs;
      xs.reserve(angular_value_cores.size() + phys_dim + 1);

      for (size_t a = 0; a < angular_value_cores.size(); a++) {
        linalg::TTEngine::Tensors cores;
        for (size_t b = 0; b < angular_value_cores.size(); b++) {
          cores.push_back(
            b == a ? angular_value_cores[a] : angular_ones_cores[b]);
        }
        for (const auto& c : spatial_ones_cores) {
          cores.push_back(c);
        }
        cores.push_back(energy_ones);
        xs.emplace_back(std::move(cores), false);
      }

      for (int64_t k = 0; k < phys_dim; k++) {
        linalg::TTEngine coord(angular_ones_cores, false);
        linalg::TTEngine spatial_k =
          linalg::TTEngine::from_dense(phys.select(-1, k).contiguous(),
            config_->rounding.eps, config_->rounding.max_rank);
        coord.kron_(spatial_k);
        coord.kron_(energy_ones);
        xs.push_back(std::move(coord));
      }

      {
        linalg::TTEngine coord(angular_ones_cores, false);
        for (const auto& c : spatial_ones_cores) {
          coord.kron_(c);
        }
        coord.kron_(energy_values);
        xs.push_back(std::move(coord));
      }

      linalg::TTEngine function_source = linalg::function_interpolate(func, xs,
        config_->cross.eps, std::nullopt, config_->cross.nswp,
        config_->cross.kick, config_->cross.max_rank, config_->cross.verbose);

      // Weight by the Jacobian mapping (broadcast across angle/energy, real
      // only in the spatial cores) -- same physical meaning as the
      // isotropic branch's spatial load vector, just applied as a Hadamard
      // product since function_source already carries angular/energy
      // dependence.
      auto basis = assemble_basis();
      auto mapping = assemble_integral_mapping();
      linalg::TTEngine mapping_full(angular_ones_cores, false);
      mapping_full.kron_(mapping);
      mapping_full.kron_(energy_ones);
      function_source *= mapping_full;
      function_source.round_(
        config_->rounding.eps / static_cast<double>(NumDim + 2),
        config_->rounding.max_rank);

      // Project onto the DG basis: a Galerkin load vector, not a pointwise
      // sample. Sandwich the spatial basis between angle/energy IDENTITY
      // operators (diagonal, self-transpose) so the whole combined object
      // can be contracted in one mm() against function_source, transposing
      // only the (non-symmetric) spatial cores -- exactly the pattern
      // TransportSolution::compute_error_terms() already uses to combine an
      // angle/energy-identity with a spatial evaluation operator.
      c10::SmallVector<int64_t, 6> angle_modes;
      for (const auto& c : angular_ones_cores) {
        angle_modes.push_back(c.size(1));
      }
      auto id_angle = linalg::TTEngine::ones(
        angle_modes, block_->get_device(), block_->get_dtype())
                        .diagonalize();
      auto id_energy = linalg::TTEngine::ones(
        {num_groups}, block_->get_device(), block_->get_dtype())
                         .diagonalize();

      linalg::TTEngine full_basis = id_angle;
      full_basis.kron_(basis);
      full_basis.kron_(id_energy);
      full_basis.transpose_();

      linalg::TTEngine projected = linalg::mm(full_basis, function_source);
      projected.round_(config_->rounding.eps, config_->rounding.max_rank);

      terms.emplace_back(std::move(projected));
    }

    linalg::TTEngine assembled =
      terms.size() == 1 ? std::move(terms[0]) : linalg::direct_sum(terms);
    assembled.round_(config_->rounding.eps, config_->rounding.max_rank);
    return linalg::State(std::move(assembled));
  }

  throw utils::runtime_error(
    "ttnte::physics::DGFirstOrderTransportBackend::assemble_source",
    "This method does not support this format yet");
}

// The face load-vector entry this ultimately contributes to (once
// inflow_ops_[face_idx] is later applied) is
//   integral_over_Ghat( R_i(xhat) * Q_inc(xhat) * |J_Gamma(xhat)| ) dAhat,
// the same "constant factors outside the integral" simplification as
// assemble_source()'s isotropic branch: Q_inc is `isotropic_strength`, one
// value per group with no spatial dependence (the `function` branch is
// rejected below precisely because a genuinely spatially-varying incident
// flux would need the same per-quadrature-point evaluation
// assemble_source()'s function branch does, which isn't implemented here
// yet), so it factors out to
//   Q_inc_g * integral_over_Ghat( R_i(xhat) * |J_Gamma(xhat)| ) dAhat.
//
// Unlike assemble_source(), that remaining spatial integral is deliberately
// NOT evaluated here. This function instead returns a raw NODAL state --
// value Q_inc_g at every tangential control point, narrowed to a single
// point along `dim` -- and defers the R_i(xhat) * |J_Gamma(xhat)| Galerkin
// projection to assemble_boundary_operators()'s inflow_ops_[face_idx],
// which is applied to this state later (mirroring exactly how the DAG's
// apply task applies coupling.boundary_op to a raw narrowed neighbor state
// for an INTERNAL face -- see configure_cpu_task.hpp). This keeps one
// single code path -- "raw nodal boundary state in, inflow/outflow operator
// (which carries the R_i x |J_Gamma| projection) applied later" -- for
// INCIDENT, REFLECTIVE, and INTERNAL faces alike, instead of a bespoke
// pre-integrated path that exists only for INCIDENT.
template<FormatType Fmt, int64_t NumDim>
linalg::State
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::assemble_incident_source(
  size_t dim, bool is_upper, const FixedSource& source)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    if (source.function.has_value()) {
      throw utils::runtime_error(
        "ttnte::physics::DGFirstOrderTransportBackend::"
        "assemble_incident_source",
        "Arbitrary-function incident boundary sources are not yet "
        "supported -- only FixedSource::isotropic_strength can be "
        "prescribed on a boundary face");
    }
    if (source.projected_isotropic_source.has_value()) {
      throw utils::runtime_error(
        "ttnte::physics::DGFirstOrderTransportBackend::"
        "assemble_incident_source",
        "FixedSource::projected_isotropic_source (a pre-projected volumetric "
        "load vector) is not supported on an INCIDENT boundary face -- only "
        "FixedSource::isotropic_strength can be prescribed on a boundary "
        "face");
    }
    if (!source.isotropic_strength.has_value()) {
      throw utils::runtime_error(
        "ttnte::physics::DGFirstOrderTransportBackend::"
        "assemble_incident_source",
        "FixedSource must define isotropic_strength for an INCIDENT "
        "boundary face");
    }

    auto options = torch::TensorOptions()
                     .device(block_->get_device())
                     .dtype(block_->get_dtype());
    int64_t num_groups = material_->get_num_groups();
    TORCH_CHECK(source.isotropic_strength->numel() == num_groups,
      "FixedSource::isotropic_strength must have one entry per energy "
      "group");

    // Angular "ones" cores: an isotropic incident flux has the same value in
    // every direction. inflow_ops_[face_idx] (built by
    // assemble_boundary_operators()) already carries the (Omega . n) < 0
    // upwind mask, so applying it to this uniform-in-angle State
    // automatically zeroes the outgoing directions -- no incoming-only mask
    // needs to be built here.
    linalg::TTEngine::Tensors angular_ones_cores;
    if constexpr (NumDim > 1) {
      if (!angular_qset_->is_tensor_product()) {
        throw utils::runtime_error(
          "ttnte::physics::DGFirstOrderTransportBackend::"
          "assemble_incident_source",
          "The angular quadrature set must be a tensor product quadrature "
          "set");
      }
      auto angular_qset =
        std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);
      const auto& quads = angular_qset->get_quads();
      angular_ones_cores.push_back(
        torch::ones({1, quads[0]->get_num_dofs(), 1, 1}, options));
      angular_ones_cores.push_back(
        torch::ones({1, quads[1]->get_num_dofs(), 1, 1}, options));
    } else {
      auto angular_qset =
        std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);
      angular_ones_cores.push_back(
        torch::ones({1, angular_qset->get_num_dofs(), 1, 1}, options));
    }

    // Spatial "ones" cores at every tangential control point: a raw NODAL
    // representation (constant everywhere for an isotropic source), NOT an
    // already basis-integrated load vector -- inflow_ops_[face_idx]'s own
    // tangential cores (built via project_boundary_mask() in
    // assemble_boundary_operators()) already perform the Galerkin
    // test-function integration when applied, exactly the way the DAG's
    // apply task applies coupling.boundary_op directly to a raw narrowed
    // neighbor State (see configure_cpu_task.hpp). The sliced `dim` itself
    // is narrowed to a single trivial point, matching inflow_op's own
    // n_mode = 1 there (the same face-extraction-column convention used for
    // INTERNAL faces).
    linalg::TTEngine::Tensors spatial_cores;
    spatial_cores.reserve(NumDim);
    for (int64_t d = 0; d < NumDim; d++) {
      int64_t n =
        (static_cast<size_t>(d) == dim) ? 1 : block_->get_ctrlpts_size(d);
      spatial_cores.push_back(torch::ones({1, n, 1, 1}, options));
    }

    linalg::TTEngine assembled(angular_ones_cores, false);
    assembled.kron_(linalg::TTEngine(spatial_cores, false));
    assembled.kron_(
      source.isotropic_strength->reshape({1, num_groups, 1, 1}).to(options));
    assembled.round_(config_->rounding.eps, config_->rounding.max_rank);

    return linalg::State(std::move(assembled));
  }

  throw utils::runtime_error(
    "ttnte::physics::DGFirstOrderTransportBackend::assemble_incident_source",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
std::tuple<typename Return<Fmt, NumDim>::Type,
  typename Return<Fmt, NumDim>::VectorType, typename Return<Fmt, NumDim>::Type>
DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_boundary_geometry(size_t dim, bool is_upper)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    double inner_eps = config_->rounding.eps / static_cast<double>(NumDim);

    // Extract the boundary block component
    auto boundary_block = block_->get_boundary(dim, is_upper);

    // Create a NumDim - 1 dimensional backend
    // TODO: Pass the decomposition of control points and weights directly to
    // the backend's cache
    auto backend = DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim - 1>(
      std::move(boundary_block), angular_qset_, *material_, *config_);

    // Compute the basis and its derivatives (for cache to compute the Jacobian)
    auto basis = backend.assemble_basis_ders()[0];

    // Compute the Jacobian
    auto jacobian = backend.assemble_jacobian();

    // Compute the integral mapping from the parametric dimension to the parent
    // element space
    const auto& options = torch::TensorOptions()
                            .device(basis.get_device())
                            .dtype(basis.get_dtype());
    linalg::TTEngine::Tensors mapping;
    mapping.reserve(NumDim - 1);
    for (int64_t i = 0; i < NumDim - 1; i++) {
      const auto& b = boundary_block->get_basis()[i];
      const auto& quad = backend.get_spatial_qset()->get_quads()[i];

      // Get the unique knots in the knot vector
      torch::Tensor unique_knots = b.get_unique_knots().to(options);

      // Compute derivatives of affine mapping
      // Note not dividing by 2 because that cancels out with the weight of the
      // quadrature
      torch::Tensor jac_1d = (unique_knots.diff().unsqueeze(1) *
                              quad->get_weights().to(options).unsqueeze(0))
                               .flatten()
                               .reshape({1, -1, 1, 1});

      // Compute rank-one mapping
      mapping.push_back(jac_1d);
    }

    // Compute the normal vector at the quadrature points
    c10::SmallVector<linalg::TTEngine, NumDim> normal;

    // Determine outward orientation
    // CAD parameterization signs alternate based on the sliced dimension.
    // dim=0 (x-face): parameters are y,z -> y x z = +x.
    // dim=1 (y-face): parameters are x,z -> x x z = -y. (Needs sign flip)
    // dim=2 (z-face): parameters are x,y -> x x y = +z.
    double orientation = block_->get_orientation() * (is_upper ? 1.0 : -1.0) *
                         ((dim % 2 != 0) ? -1.0 : 1.0);

    if constexpr (NumDim == 2) {
      // 2-D problem -> 1-D boundary
      assert(jacobian[2][0].get_cores().empty());
      normal.push_back(orientation * jacobian[1][0]);
      normal.push_back(-orientation * jacobian[0][0]);

    } else if constexpr (NumDim == 3) {
      // 3-D problem -> 2-D boundary
      // Compute the cross product of the tangent vectors
      // tu = (dx/du, dy/du, dz/du)
      // tv = (dx/dv, dy/dv, dz/dv)
      // n = tu x tv
      auto dx_du = jacobian[0][0];
      auto dx_dv = jacobian[0][1];
      auto dy_du = jacobian[1][0];
      auto dy_dv = jacobian[1][1];
      auto dz_du = jacobian[2][0];
      auto dz_dv = jacobian[2][1];

      // Cross product arithmetic (TT Hadamard products and TT subtraction)
      auto nx = orientation * ((dy_du * dz_dv) - (dz_du * dy_dv))
                                .round(inner_eps, config_->rounding.max_rank);
      auto ny = orientation * ((dz_du * dx_dv) - (dx_du * dz_dv))
                                .round(inner_eps, config_->rounding.max_rank);
      auto nz = orientation * ((dx_du * dy_dv) - (dy_du * dx_dv))
                                .round(inner_eps, config_->rounding.max_rank);

      // Add to the normal vector
      normal.push_back(std::move(nx));
      normal.push_back(std::move(ny));
      normal.push_back(std::move(nz));
    }

    return std::make_tuple(
      std::move(basis), std::move(normal), ReturnType(std::move(mapping)));
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_boundary_geometry",
    "This method is only valid for FormatType::TENSOR_TRAIN with NumDim > 1");
}

template<FormatType Fmt, int64_t NumDim>
std::tuple<linalg::Operator, linalg::Operator, linalg::Operator>
DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_boundary_operators(size_t dim, bool is_upper)
{
  // Return nothing if the boundary is degenerate
  if (block_->get_boundary_info(dim, is_upper).get_type() ==
      BoundaryType::DEGENERATE) {
    return std::make_tuple(
      linalg::Operator(), linalg::Operator(), linalg::Operator());
  }

  // Partial (specular) albedo: only meaningful on a REFLECTIVE face. This is
  // the SINGLE place the albedo enters the transport operator -- the
  // REFLECTIVE inflow operator B_refl returned below is scaled by it, so
  // every consumer of the returned inflow operator (the assembler's LHS,
  // `lhs -= inflow_op`, and its stored `inflow_ops_`) sees the same
  // albedo * B_refl. albedo == 1 skips the scaling entirely (bit-for-bit
  // identical to plain REFLECTIVE); albedo == 0 gives a zero inflow
  // operator (VACUUM to roundoff).
  const double albedo = block_->get_boundary_info(dim, is_upper).albedo();
  if (albedo != 1.0 && block_->get_boundary_info(dim, is_upper).get_type() !=
                         BoundaryType::REFLECTIVE) {
    throw utils::runtime_error("ttnte::physics::DGFirstOrderTransportBackend::"
                               "assemble_boundary_operators",
      "Boundary face (dim=" + std::to_string(dim) +
        ", is_upper=" + (is_upper ? "true" : "false") + ") has albedo " +
        std::to_string(albedo) +
        " != 1 but is not BoundaryType::REFLECTIVE -- an albedo is only "
        "valid on REFLECTIVE faces");
  }

  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    auto [basis, normal, mapping] = assemble_boundary_geometry(dim, is_upper);
    const auto& options = torch::TensorOptions()
                            .device(basis.get_device())
                            .dtype(basis.get_dtype());

    // Get the outflow and inflow boundary operators
    const auto condition = block_->get_boundary_info(dim, is_upper).get_type();
    auto B_out = assemble_outflow_boundary_operator(basis, normal, mapping);
    auto B_in =
      assemble_inflow_boundary_operator(basis, normal, mapping, condition);

    // Create a core for the new dimension with a 1.0 in the spot of
    // the interpolatory control point evaluation
    size_t target_idx = 2 + dim;
    int64_t num_ctrlpts = block_->get_ctrlpts_size(dim);
    int64_t face_idx = is_upper ? num_ctrlpts - 1 : 0;

    // Outflow / reflective inflow: full spatial core (n_mode = num_ctrlpts)
    auto target_core = torch::zeros({num_ctrlpts, num_ctrlpts}, options);
    target_core[face_idx][face_idx] = 1.0;
    target_core.unsqueeze_(0).unsqueeze_(-1);

    // INTERNAL / INCIDENT inflow: face-extraction column (n_mode = 1,
    // m_mode = num_ctrlpts) -- both apply their inflow operator to a State
    // already narrowed to a single point in `dim` (a NeighborCoupling's
    // recv_buffer for INTERNAL, a user-supplied incident-flux State for
    // INCIDENT), folding it into the full local basis.
    torch::Tensor inflow_target_core;
    if (condition == BoundaryType::INTERNAL ||
        condition == BoundaryType::INCIDENT) {
      inflow_target_core = torch::zeros({num_ctrlpts, 1}, options);
      inflow_target_core[face_idx][0] = 1.0;
      inflow_target_core.unsqueeze_(0).unsqueeze_(-1);
    } else {
      inflow_target_core = target_core;
    }

    // Create an identity core for energy
    auto energy = torch::eye(material_->get_num_groups(), options)
                    .unsqueeze_(0)
                    .unsqueeze_(-1);

    auto inject_basis_and_energy = [&](const linalg::TTEngine& B,
                                     const torch::Tensor& tc) {
      // Get the cores
      auto B_cores = B.get_cores();
      B_cores.reserve(B_cores.size() + 2);

      // Inject the core either in between ranks or at the end of the TT
      if (target_idx < NumDim + 1) {
        int64_t r = B_cores[target_idx].size(0);
        B_cores.insert(B_cores.begin() + target_idx,
          torch::eye(r, options).reshape({r, 1, 1, r}) * tc);

      } else {
        B_cores.push_back(tc);
      }

      // Append an energy core
      B_cores.push_back(energy);

      return linalg::TTEngine(std::move(B_cores), false);
    };

    // Build the current-reduction operator: the (Omega . n)_+ outflow
    // upwind mask folded with the angular quadrature weights (reusing
    // apply_angular_weights(), the same helper assemble_scattering_kernel()
    // uses), then its tangential (face-quadrature) cores projected onto the
    // DG basis via project_boundary_mask() -- the exact same treatment
    // B_out/B_in get below. This projection is required, not optional: for
    // NumDim > 1, assemble_upwind_mask()'s tangential cores live in a
    // face-quadrature-point basis, not the DOF/control-point basis
    // inject_basis_and_energy()'s identity cores use below -- kron-ing the
    // two together without first reconciling them into the same basis would
    // double-represent that spatial axis (once from the mask, once from the
    // identity), producing an operator with too many TT-cores. Applying the
    // result to a boundary-narrowed angular-flux state and reducing the
    // angular core(s) away yields the outgoing partial current at this
    // face, spatially(+energy) resolved -- used only as a Schwarz
    // convergence indicator (LocalSolver::postsolve()), not part of the
    // actual PDE operator. Only needed for INTERNAL boundaries (the only
    // case a NeighborCoupling exists for).
    linalg::Operator current_op;
    if (condition == BoundaryType::INTERNAL) {
      linalg::TTEngine current_mask =
        assemble_upwind_mask(normal, /*is_outflow=*/true);
      current_mask = apply_angular_weights(current_mask, {0, 1});
      current_mask =
        project_boundary_mask(std::move(current_mask), basis, mapping);

      // Trivial 1x1 "identity" for the narrowed dim -- unlike B_out/B_in's
      // target_core (which selects the boundary index out of the full,
      // non-narrowed state), this operator is applied to a state already
      // narrowed to size 1 along `dim` (see LocalSolver::postsolve()), so no
      // selection is needed, just a pass-through.
      auto trivial_core = torch::ones({1, 1, 1, 1}, options);

      current_mask = inject_basis_and_energy(current_mask, trivial_core);
      // Angular cores are left reduced (m=1, from apply_angular_weights) --
      // deliberately NOT diagonalized like B_out/B_in below -- since
      // applying this operator is meant to integrate the angular dependence
      // away, not preserve it.
      current_op = linalg::Operator(std::move(current_mask));
    }

    // Make final trains
    B_out = inject_basis_and_energy(std::move(B_out), target_core)
              .diagonalize({0, 1});

    if (condition != BoundaryType::VACUUM) {
      B_in = inject_basis_and_energy(std::move(*B_in), inflow_target_core);

      if (condition == BoundaryType::INTERNAL ||
          condition == BoundaryType::INCIDENT) {
        B_in->diagonalize_({0, 1});
      }
      if (condition == BoundaryType::REFLECTIVE && albedo != 1.0) {
        *B_in *= albedo;
      }
      return std::make_tuple(linalg::Operator(std::move(B_out)),
        linalg::Operator(std::move(*B_in)), std::move(current_op));
    }

    return std::make_tuple(linalg::Operator(std::move(B_out)),
      linalg::Operator(), linalg::Operator());

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    const auto& options =
      torch::TensorOptions().device(block_->get_device()).dtype(config_->dtype);
    double orientation = block_->get_orientation() * (is_upper ? 1.0 : -1.0);

    linalg::TTEngine normal(
      {torch::tensor({orientation}, options).reshape({1, 1, 1, 1})});

    const auto condition = block_->get_boundary_info(dim, is_upper).get_type();
    auto B_out = assemble_outflow_boundary_operator({}, {normal}, {});
    auto B_in = assemble_inflow_boundary_operator({}, {normal}, {}, condition);

    // Create spatial core
    int64_t num_ctrlpts = block_->get_ctrlpts_size(dim);
    int64_t face_idx = is_upper ? num_ctrlpts - 1 : 0;

    // Outflow / reflective inflow: full spatial core (n_mode = num_ctrlpts)
    auto target_core = torch::zeros({num_ctrlpts, num_ctrlpts}, options);
    target_core[face_idx][face_idx] = 1.0;
    target_core.unsqueeze_(0).unsqueeze_(-1);

    // INTERNAL / INCIDENT inflow: face-extraction column (n_mode = 1,
    // m_mode = num_ctrlpts) -- see the NumDim > 1 branch above for the full
    // rationale.
    torch::Tensor inflow_target_core;
    if (condition == BoundaryType::INTERNAL ||
        condition == BoundaryType::INCIDENT) {
      inflow_target_core = torch::zeros({num_ctrlpts, 1}, options);
      inflow_target_core[face_idx][0] = 1.0;
      inflow_target_core.unsqueeze_(0).unsqueeze_(-1);
    } else {
      inflow_target_core = target_core;
    }

    // Create identity core for energy groups
    auto energy = torch::eye(material_->get_num_groups(), options)
                    .unsqueeze_(0)
                    .unsqueeze_(-1);

    auto inject_basis_and_energy = [&](const linalg::TTEngine& B,
                                     const torch::Tensor& tc) {
      auto B_cores = B.get_cores();
      B_cores.reserve(B_cores.size() + 2);

      // Append space and energy cores
      B_cores.push_back(tc);
      B_cores.push_back(energy);

      return linalg::TTEngine(std::move(B_cores), false);
    };

    // Build the current-reduction operator -- see the NumDim > 1 branch
    // above for the full rationale; only needed for INTERNAL boundaries.
    linalg::Operator current_op;
    if (condition == BoundaryType::INTERNAL) {
      linalg::TTEngine current_mask =
        assemble_upwind_mask({normal}, /*is_outflow=*/true);
      current_mask = apply_angular_weights(current_mask, {0});

      c10::SmallVector<int64_t, 6> rest_modes = {
        1, material_->get_num_groups()};
      linalg::TTEngine identity_op = linalg::TTEngine::ones(
        rest_modes, current_mask.get_device(), current_mask.get_dtype())
                                       .diagonalize();
      current_op = linalg::Operator(current_mask.kron(identity_op));
    }

    // Make final trains
    B_out =
      inject_basis_and_energy(std::move(B_out), target_core).diagonalize({0});

    if (condition != BoundaryType::VACUUM) {
      B_in = inject_basis_and_energy(std::move(*B_in), inflow_target_core);

      if (condition == BoundaryType::INTERNAL ||
          condition == BoundaryType::INCIDENT) {
        B_in->diagonalize_({0});
      }
      if (condition == BoundaryType::REFLECTIVE && albedo != 1.0) {
        *B_in *= albedo;
      }
      return std::make_tuple(linalg::Operator(std::move(B_out)),
        linalg::Operator(std::move(*B_in)), std::move(current_op));
    }

    return std::make_tuple(linalg::Operator(std::move(B_out)),
      linalg::Operator(), linalg::Operator());
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_boundary_operators",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_outflow_boundary_operator(const ReturnType& basis,
  const typename Return<Fmt, NumDim>::VectorType& normal,
  const ReturnType& mapping)
{
  return assemble_interface_boundary_operator(basis, normal, mapping, true);
}

template<FormatType Fmt, int64_t NumDim>
std::optional<typename Return<Fmt, NumDim>::Type>
DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_inflow_boundary_operator(const ReturnType& basis,
  const typename Return<Fmt, NumDim>::VectorType& normal,
  const ReturnType& mapping, const BoundaryType condition)
{
  if (condition == BoundaryType::VACUUM) {
    return std::nullopt;
  }

  if (condition == BoundaryType::REFLECTIVE) {
    if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
      // Get the outflow magnitudes to reuse their calculations
      auto B_cores =
        assemble_outflow_boundary_operator(basis, normal, mapping).get_cores();

      auto device = B_cores[0].device();
      auto dtype = B_cores[0].scalar_type();

      // Extract quadrature points to build permutation vectors
      auto angular_qset =
        std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);
      auto points = angular_qset->get_factored_points();

      torch::Tensor mu = points[0].to(basis.get_device());
      torch::Tensor gamma = points[1].to(basis.get_device());

      // Initialize permutations as identity mappings
      torch::Tensor perm_0 = torch::arange(
        mu.size(0), torch::TensorOptions().device(device).dtype(torch::kLong));
      torch::Tensor perm_1 = torch::arange(gamma.size(0),
        torch::TensorOptions().device(device).dtype(torch::kLong));

      // Determine the physical axis of reflection dynamically
      // Since the boundary is axis-aligned, exactly one physical component
      // of the `normal` TT vector will be dominant/non-zero.
      size_t space_dim = 0;
      size_t space_dims = cache_.get_ctrlpts().size();
      double max_magnitude = -1.0;

      for (size_t i = 0; i < space_dims; ++i) {
        double comp_max = 0.0;
        // Evaluate the max absolute value inside the TT cores to find the
        // non-zero normal component
        for (const torch::Tensor& core : normal[i]) {
          comp_max = std::max(comp_max, core.abs().max().item<double>());
        }

        if (comp_max > max_magnitude) {
          max_magnitude = comp_max;
          space_dim = i;
        }
      }

      // Axis-aligned reflection targets based on the discovered physical
      // dimension
      if (space_dim == 0) {
        // X-reflection: cos(g) -> -cos(g), sin(g) -> sin(g). (Changes Core 1)
        auto cos = torch::cos(gamma);
        auto sin = torch::sin(gamma);
        auto dist = torch::square(cos.unsqueeze(1) + cos.unsqueeze(0)) +
                    torch::square(sin.unsqueeze(1) - sin.unsqueeze(0));
        perm_1 = torch::argmin(dist, 1);

      } else if (space_dim == 1) {
        // Y-reflection: cos(g) -> cos(g), sin(g) -> -sin(g). (Changes Core 1)
        auto cos = torch::cos(gamma);
        auto sin = torch::sin(gamma);
        auto dist = torch::square(cos.unsqueeze(1) - cos.unsqueeze(0)) +
                    torch::square(sin.unsqueeze(1) + sin.unsqueeze(0));
        perm_1 = torch::argmin(dist, 1);

      } else if (space_dim == 2) {
        // Z-reflection: mu -> -mu. (Changes Core 0)
        auto dist = torch::abs(mu.unsqueeze(1) + mu.unsqueeze(0));
        perm_0 = torch::argmin(dist, 1);
      }

      // Lambda helper to transform a vector core (r_left, N, r_right) into a
      // permuted diagonal operator core (r_left, N, N, r_right)
      auto transform_core = [](const torch::Tensor& core,
                              const torch::Tensor& perm) {
        int64_t r_left = core.size(0);
        int64_t N = core.size(1);
        int64_t r_right = core.size(2);

        auto reflected_core =
          torch::zeros({r_left, N, N, r_right}, core.options());

        for (int64_t k = 0; k < N; ++k) {
          int64_t k_star = perm[k].item<int64_t>();
          reflected_core.index_put_(
            {torch::indexing::Slice(), k, k_star, torch::indexing::Slice()},
            core.index(
              {torch::indexing::Slice(), k_star, 0, torch::indexing::Slice()}));
        }
        return reflected_core;
      };

      // 5. Transform the two angular cores (0 and 1)
      B_cores[0] = transform_core(B_cores[0], perm_0);
      B_cores[1] = transform_core(B_cores[1], perm_1);

      return linalg::TTEngine(std::move(B_cores), false);

    } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
      // Get outflow boundary operator
      auto core = assemble_outflow_boundary_operator(basis, normal, mapping)[0];

      // Get ordinates
      assert(!angular_qset_->is_tensor_product());
      auto mu = std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_)
                  ->get_points();
      int64_t num_ordinates = mu.numel();

      // Pure 1-D Angular Quadrature: Flip mu -> -mu (Changes Core 0)
      auto dist = torch::abs(mu.unsqueeze(1) + mu.unsqueeze(0));
      torch::Tensor perm = torch::argmin(dist, 1);

      auto reflected_core =
        torch::zeros({1, num_ordinates, num_ordinates, 1}, core.options());
      for (int64_t k = 0; k < num_ordinates; k++) {
        int64_t k_star = perm[k].item<int64_t>();
        reflected_core.index_put_(
          {torch::indexing::Slice(), k, k_star, torch::indexing::Slice()},
          core.index(
            {torch::indexing::Slice(), k_star, 0, torch::indexing::Slice()}));
      }

      return linalg::TTEngine({reflected_core}, false);
    }
  }

  if (condition == BoundaryType::INTERNAL ||
      condition == BoundaryType::INCIDENT) {
    return assemble_interface_boundary_operator(basis, normal, mapping, false);
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_inflow_boundary_operator",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::assemble_upwind_mask(
  const typename Return<Fmt, NumDim>::VectorType& normal, bool is_outflow)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    // Get the ordinates in TT format
    auto ordinates = assemble_ordinates();

    // Compute dot(ordinates, normal) in TT format
    linalg::TTEngine ndo = ordinates[0].kron(normal[0]);
    for (size_t i = 1; i < NumDim; i++) {
      ndo += ordinates[i].kron(normal[i]);
    }
    ndo.round_(config_->rounding.eps, config_->rounding.max_rank);

    // Compute abs(dot(ordinates, normal))
    auto ando = apply_mult_separable(
      ndo, [](const torch::Tensor& tensor) { return torch::abs(tensor); },
      config_->rounding, config_->cross, config_->max_dense_size);

    // Compute the outflow/inflow dot(ordinates, normal)
    return static_cast<double>(0.5) *
           (ando + (is_outflow ? ndo : -ndo))
             .round(config_->rounding.eps, config_->rounding.max_rank);

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    // Get ordinates
    torch::Tensor ordinates = assemble_ordinates()[0][0];

    // Compute dot(ordinates, normal)
    torch::Tensor ndo = ordinates * normal[0][0];

    return linalg::TTEngine(
      {torch::clamp(is_outflow ? ndo : -ndo, 0).reshape({1, -1, 1, 1})}, false);
  }

  throw utils::runtime_error(
    "ttnte::physics::DIGAFirstOrderTransportBackend::assemble_upwind_mask",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type
DGFirstOrderTransportBackend<cad::Patch, Fmt, NumDim>::project_boundary_mask(
  linalg::TTEngine mask, const ReturnType& basis, const ReturnType& mapping)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    // Apply the basis
    for (size_t i = 0; i < NumDim - 1; i++) {
      auto& core = mask[i + 2];
      int64_t rl_b = basis[i].size(0);
      int64_t m = basis[i].size(1);
      int64_t n = basis[i].size(2);

      core = torch::einsum("abcd,ebfg->aebfdg", {core, basis[i]})
               .reshape({core.size(0) * rl_b, m, n, -1});
    }
    mask.round_(config_->rounding.eps, config_->rounding.max_rank);

    // Compute the outer product with the mapped basis
    for (size_t i = 0; i < NumDim - 1; i++) {
      auto& core = mask[i + 2];
      int64_t rl_b = basis[i].size(0);
      int64_t m = core.size(2);
      int64_t n = basis[i].size(2);

      auto mapped_basis = basis[i] * mapping[i];

      core = torch::einsum("abcd,ebfg->aecfdg", {core, mapped_basis})
               .reshape({core.size(0) * rl_b, m, n, -1});
    }
    mask.round_(config_->rounding.eps, config_->rounding.max_rank);

    return mask;
  }

  throw utils::runtime_error(
    "ttnte::physics::DIGAFirstOrderTransportBackend::project_boundary_mask",
    "This method is only valid for FormatType::TENSOR_TRAIN with NumDim > 1");
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_interface_boundary_operator(const ReturnType& basis,
  const typename Return<Fmt, NumDim>::VectorType& normal,
  const ReturnType& mapping, bool is_outflow)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    linalg::TTEngine B = assemble_upwind_mask(normal, is_outflow);
    return project_boundary_mask(std::move(B), basis, mapping);

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    return assemble_upwind_mask(normal, is_outflow);
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_interface_boundary_operator",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
linalg::Operator DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_balance_functional(const torch::Tensor& energy_matrix,
  double eps, int64_t max_rank)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    auto options = torch::TensorOptions()
                     .device(block_->get_device())
                     .dtype(block_->get_dtype());

    // Angular factor: bare quadrature weights, NOT expanded/redistributed
    // the way assemble_angular_integral() does for fission's isotropic
    // reinjection -- this collapses angle to size 1 directly when applied
    // via mv(), since the balance quantity is a scalar functional, not a
    // reusable redistribution operator.
    linalg::TTEngine::Tensors angular_cores;
    if constexpr (NumDim > 1) {
      if (!angular_qset_->is_tensor_product()) {
        throw utils::runtime_error(
          "ttnte::physics::DGFirstOrderTransportBackend::"
          "assemble_balance_functional",
          "The angular quadrature set must be a tensor product quadrature set");
      }
      auto angular_qset =
        std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);
      const auto& quads = angular_qset->get_quads();
      angular_cores.push_back(
        quads[0]->get_weights().to(options).reshape({1, 1, -1, 1}));
      angular_cores.push_back(
        quads[1]->get_weights().to(options).reshape({1, 1, -1, 1}));
    } else {
      auto angular_qset =
        std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);
      angular_cores.push_back(
        angular_qset->get_weights().to(options).reshape({1, 1, -1, 1}));
    }
    linalg::TTEngine functional(angular_cores, false);

    // Spatial factor: a one-sided load vector -- the same construction
    // assemble_source()'s isotropic branch uses (mm(basis^T, mapping), NOT
    // the bilinear mm(basis^T, mapped_basis) mass matrix
    // assemble_fission_operator()/assemble_scatter_operator() need for their
    // reusable DOF-space output) -- transposed into a reduction (m=1)
    // instead of an injection (n=1).
    auto basis = assemble_basis();
    auto mapping = assemble_integral_mapping();
    linalg::TTEngine spatial = linalg::mm(basis.transpose(), mapping);
    spatial.transpose_();
    spatial.round_(eps, max_rank);
    functional.kron_(spatial);

    // Energy factor: the supplied weight/transfer matrix, preserved (not
    // reduced) so the group axis survives in the output.
    int64_t num_groups = material_->get_num_groups();
    functional.kron_(
      energy_matrix.to(options).reshape({1, num_groups, num_groups, 1}));

    functional.round_(eps, max_rank);
    return linalg::Operator(std::move(functional));
  }

  throw utils::runtime_error(
    "ttnte::physics::DGFirstOrderTransportBackend::assemble_balance_functional",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
linalg::Operator DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_moment_projector(int64_t order)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN) {
    if (order != 1) {
      throw utils::runtime_error(
        "ttnte::physics::DGFirstOrderTransportBackend::"
        "assemble_moment_projector",
        "Only order = 1 (scalar flux + current) is currently supported");
    }

    auto options = torch::TensorOptions()
                     .device(block_->get_device())
                     .dtype(block_->get_dtype());

    // Identity over space + energy: a pure pass-through, NOT the DG-basis
    // mass matrix assemble_scattering_kernel()'s `spatial` argument builds --
    // a moment is defined purely by integrating angle, so the state's own
    // spatial/energy representation must survive unchanged.
    c10::SmallVector<int64_t, 6> rest_modes;
    for (int64_t d = 0; d < NumDim; d++) {
      rest_modes.push_back(block_->get_ctrlpts_size(static_cast<size_t>(d)));
    }
    rest_modes.push_back(material_->get_num_groups());
    linalg::TTEngine identity_op = linalg::TTEngine::ones(
      rest_modes, block_->get_device(), block_->get_dtype())
                                     .diagonalize();

    // General (non-orthogonal-safe) projector onto span{b_i}: P = sum_ij
    // b_i (G^-1)_ij <b_j, .>_w, where G_ij = <b_i,b_j>_w is the (small,
    // n_moments x n_moments) Gram matrix of the moment basis under the
    // quadrature-weighted inner product. NOTE: the b_i are NOT assumed
    // mutually orthogonal -- e.g. for NumDim > 1 the polar quadrature is
    // half-range (mu in [0,1], see ProductQuadrature's hemisphere-reflection
    // symmetry), so <1,mu>_w != 0 and P0/P1z are genuinely non-orthogonal; a
    // naive per-term |beta_i><beta_i| sum (valid only when orthogonal) is
    // measurably wrong there. The quadrature measure factors as
    // w = w_polar * w_azim, so G_ij = Gp_ij * Ga_ij factors the same way,
    // letting each (i,j) term stay separable across the two angular cores.
    std::vector<linalg::TTEngine> terms;

    if constexpr (NumDim > 1) {
      if (!angular_qset_->is_tensor_product()) {
        throw utils::runtime_error(
          "ttnte::physics::DGFirstOrderTransportBackend::"
          "assemble_moment_projector",
          "The angular quadrature set must be a tensor product quadrature set");
      }
      auto angular_qset =
        std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);
      const auto& quads = angular_qset->get_quads();
      auto w_polar = quads[0]->get_weights().to(options);
      auto w_azim = quads[1]->get_weights().to(options);
      int64_t nm = quads[0]->get_num_dofs();
      int64_t ng = quads[1]->get_num_dofs();

      // b_polar[i]/b_azim[i]: P0 (1, 1), P1z (mu, 1), P1x (sqrt(1-mu^2),
      // cos), P1y (sqrt(1-mu^2), sin) -- P1x/y/z reuse assemble_ordinates()'s
      // cached Omega_x/y/z (index 0/1/2 respectively), each already factored
      // exactly along (polar, azimuthal).
      const auto& ordinates = assemble_ordinates();
      torch::Tensor one_polar = torch::ones({nm}, options);
      torch::Tensor one_azim = torch::ones({ng}, options);
      std::array<torch::Tensor, 4> b_polar = {one_polar,
        ordinates[2][0].reshape({-1}), ordinates[0][0].reshape({-1}),
        ordinates[1][0].reshape({-1})};
      std::array<torch::Tensor, 4> b_azim = {one_azim,
        ordinates[2][1].reshape({-1}), ordinates[0][1].reshape({-1}),
        ordinates[1][1].reshape({-1})};

      const int64_t n = static_cast<int64_t>(b_polar.size());
      torch::Tensor Gp = torch::empty({n, n}, options);
      torch::Tensor Ga = torch::empty({n, n}, options);
      for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
          Gp[i][j] = (w_polar * b_polar[i] * b_polar[j]).sum();
          Ga[i][j] = (w_azim * b_azim[i] * b_azim[j]).sum();
        }
      }
      torch::Tensor Ginv = torch::linalg_inv(Gp * Ga);

      for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
          linalg::TTEngine::Tensors term_cores;
          term_cores.push_back(
            (Ginv[i][j] * torch::outer(b_polar[i], w_polar * b_polar[j]))
              .reshape({1, nm, nm, 1}));
          term_cores.push_back(torch::outer(b_azim[i], w_azim * b_azim[j])
              .reshape({1, ng, ng, 1}));
          linalg::TTEngine term(term_cores, false);
          term.kron_(identity_op);
          terms.push_back(std::move(term));
        }
      }

    } else {
      auto angular_qset =
        std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);
      auto w_polar = angular_qset->get_weights().to(options);
      int64_t nm = angular_qset->get_num_dofs();

      // b_polar[i]: P0 (1), P1z (mu) -- mu reuses assemble_ordinates()'s
      // single cached ordinate (a slab has only the polar cosine).
      const auto& ordinates = assemble_ordinates();
      torch::Tensor one_polar = torch::ones({nm}, options);
      std::array<torch::Tensor, 2> b_polar = {
        one_polar, ordinates[0][0].reshape({-1})};

      const int64_t n = static_cast<int64_t>(b_polar.size());
      torch::Tensor G = torch::empty({n, n}, options);
      for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
          G[i][j] = (w_polar * b_polar[i] * b_polar[j]).sum();
        }
      }
      torch::Tensor Ginv = torch::linalg_inv(G);

      for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
          linalg::TTEngine::Tensors term_cores;
          term_cores.push_back(
            (Ginv[i][j] * torch::outer(b_polar[i], w_polar * b_polar[j]))
              .reshape({1, nm, nm, 1}));
          linalg::TTEngine term(term_cores, false);
          term.kron_(identity_op);
          terms.push_back(std::move(term));
        }
      }
    }

    // Exact sum -- no truncation needed, the combined bond rank is exactly
    // terms.size() (n_moments^2) by construction.
    return linalg::Operator(linalg::direct_sum(terms));
  }

  throw utils::runtime_error(
    "ttnte::physics::DGFirstOrderTransportBackend::assemble_moment_projector",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
linalg::Operator DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_leakage_functional(size_t dim, bool is_upper,
  bool is_outflow, bool narrowed_input, double eps, int64_t max_rank)
{
  if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim > 1) {
    auto [basis, normal, mapping] = assemble_boundary_geometry(dim, is_upper);
    auto options = torch::TensorOptions()
                     .device(basis.get_device())
                     .dtype(basis.get_dtype());

    if (!angular_qset_->is_tensor_product()) {
      throw utils::runtime_error(
        "ttnte::physics::DGFirstOrderTransportBackend::"
        "assemble_leakage_functional",
        "The angular quadrature set must be a tensor product quadrature set");
    }
    auto angular_qset =
      std::static_pointer_cast<math::ProductQuadrature>(angular_qset_);
    const auto& quads = angular_qset->get_quads();

    // Raw (Omega . n)_+/- mask (curved-boundary aware, via
    // assemble_boundary_geometry()'s pointwise normal) -- a value-per-
    // direction/value-per-tangential-quadrature-point representation
    // (n=1 throughout every core). Transpose into a reduction (m=1)
    // convention before weighting/projecting.
    linalg::TTEngine mask = assemble_upwind_mask(normal, is_outflow);
    mask.transpose_();

    // Weight the angular cores by the quadrature weights -- safe now that
    // the 'n' axis holds the angular samples (pre-transpose this would
    // broadcast into an (nm x nm) outer product instead of a reweight).
    mask[0] =
      mask[0] * quads[0]->get_weights().to(options).reshape({1, 1, -1, 1});
    mask[1] =
      mask[1] * quads[1]->get_weights().to(options).reshape({1, 1, -1, 1});

    // One-sided tangential reduction: contract each tangential core's
    // quadrature-point axis against the Jacobian/quadrature-weighted mapped
    // basis -- only the trial-side half of project_boundary_mask()'s
    // bilinear (test + trial) projection, since a pure scalar reduction
    // doesn't need the test-side reprojection back onto the control-point
    // basis the real B_out/B_in/current_op operators require.
    for (size_t i = 0; i < NumDim - 1; i++) {
      auto& core = mask[i + 2];
      int64_t rl = core.size(0);
      int64_t rl_b = basis[i].size(0);
      int64_t ctrlpts = basis[i].size(2);
      auto mapped_basis = basis[i] * mapping[i];

      core = torch::einsum("abcd,ecfg->aebfdg", {core, mapped_basis})
               .reshape({rl * rl_b, 1, ctrlpts, -1});
    }
    mask.round_(eps, max_rank);

    // Insert the missing `dim`-axis core: a face-extraction row (m=1,
    // n=num_ctrlpts, a single 1 at this face's boundary control point) --
    // ndo/mask never carried a core for `dim` itself (only the tangential
    // dims), so this is the reduction-functional analog of
    // assemble_boundary_operators()'s inject_basis_and_energy() lambda,
    // which inserts its own (m=n=num_ctrlpts) target_core at the same
    // target_idx = 2 + dim position.
    {
      torch::Tensor face_core_content;
      if (narrowed_input) {
        // Applied to a State already narrowed to a single point along `dim`
        // (e.g. assemble_incident_source()'s output) -- a trivial n=1
        // pass-through, matching inflow_target_core's convention for
        // INTERNAL/INCIDENT inflow in assemble_boundary_operators().
        face_core_content = torch::ones({1, 1, 1, 1}, options);
      } else {
        int64_t num_ctrlpts = block_->get_ctrlpts_size(dim);
        int64_t face_idx = is_upper ? num_ctrlpts - 1 : 0;
        face_core_content = torch::zeros({1, 1, num_ctrlpts, 1}, options);
        face_core_content.index_put_({0, 0, face_idx, 0}, 1.0);
      }

      auto cores = mask.get_cores();
      size_t target_idx = 2 + dim;
      if (target_idx < cores.size() + 1) {
        // A hardcoded rank-1 core here would silently corrupt the TT chain
        // whenever the true bond rank at this insertion point isn't 1 (only
        // guaranteed when target_idx lands at the very end, i.e. dim ==
        // NumDim - 1, since a canonical TT's last core always has r_right =
        // 1 -- appending is always safe, but inserting into the middle of
        // the chain is not). Mirror inject_basis_and_energy()'s own fix for
        // exactly this: read the actual rank at the insertion point and
        // broadcast the face content across an r x r identity, so the
        // inserted core's r_left/r_right match its new neighbors exactly.
        // target_idx == cores.size() is the append case (no "next" core to
        // read from) -- take the rank from the current last core's r_right
        // instead.
        int64_t r = target_idx < cores.size() ? cores[target_idx].size(0)
                                              : cores.back().size(3);
        torch::Tensor face_core =
          torch::eye(r, options).reshape({r, 1, 1, r}) * face_core_content;
        cores.insert(cores.begin() + static_cast<int64_t>(target_idx),
          std::move(face_core));
      } else {
        cores.push_back(std::move(face_core_content));
      }
      mask = linalg::TTEngine(std::move(cores), false);
    }

    // Energy factor: identity -- leakage is a pure particle current, not
    // XS-weighted.
    int64_t num_groups = material_->get_num_groups();
    mask.kron_(
      torch::eye(num_groups, options).reshape({1, num_groups, num_groups, 1}));
    mask.round_(eps, max_rank);

    return linalg::Operator(std::move(mask));

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    auto options =
      torch::TensorOptions().device(block_->get_device()).dtype(config_->dtype);
    double orientation = block_->get_orientation() * (is_upper ? 1.0 : -1.0);
    linalg::TTEngine normal(
      {torch::tensor({orientation}, options).reshape({1, 1, 1, 1})});

    linalg::TTEngine mask = assemble_upwind_mask({normal}, is_outflow);
    mask.transpose_();

    auto angular_qset =
      std::static_pointer_cast<math::QuadratureSet1D>(angular_qset_);
    mask[0] =
      mask[0] * angular_qset->get_weights().to(options).reshape({1, 1, -1, 1});
    mask.round_(eps, max_rank);

    // Insert the missing spatial (`dim`) axis: a face-extraction row (m=1,
    // n=num_ctrlpts), matching the NumDim > 1 branch above -- assemble_
    // upwind_mask() for NumDim == 1 is angular-only, no spatial core at all.
    if (narrowed_input) {
      mask.kron_(torch::ones({1, 1, 1, 1}, options));
    } else {
      int64_t num_ctrlpts = block_->get_ctrlpts_size(dim);
      int64_t face_idx = is_upper ? num_ctrlpts - 1 : 0;
      auto face_core = torch::zeros({1, 1, num_ctrlpts, 1}, options);
      face_core.index_put_({0, 0, face_idx, 0}, 1.0);
      mask.kron_(face_core);
    }

    int64_t num_groups = material_->get_num_groups();
    mask.kron_(
      torch::eye(num_groups, options).reshape({1, num_groups, num_groups, 1}));
    mask.round_(eps, max_rank);

    return linalg::Operator(std::move(mask));
  }

  throw utils::runtime_error(
    "ttnte::physics::DGFirstOrderTransportBackend::assemble_leakage_functional",
    "This method does not support this format yet");
}

// Explicit template instantiations for targeting configurations
template class DGFirstOrderTransportBackend<cad::Patch, FormatType::DENSE, 1>;
template class DGFirstOrderTransportBackend<cad::Patch, FormatType::DENSE, 2>;
template class DGFirstOrderTransportBackend<cad::Patch, FormatType::DENSE, 3>;
template class DGFirstOrderTransportBackend<cad::Patch,
  FormatType::TENSOR_TRAIN, 1>;
template class DGFirstOrderTransportBackend<cad::Patch,
  FormatType::TENSOR_TRAIN, 2>;
template class DGFirstOrderTransportBackend<cad::Patch,
  FormatType::TENSOR_TRAIN, 3>;

} // namespace ttnte::physics::backends
