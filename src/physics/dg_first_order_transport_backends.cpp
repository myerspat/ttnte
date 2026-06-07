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
  Func func, const ttnte::physics::TTConfig& rounding,
  const ttnte::physics::CrossConfig& cross, int max_dense_size)
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
      const int32_t num_terms = (NumDim - 1) * l + 1;
      const double l_f = l;

      linalg::TTEngine Yl(linalg::TTEngine::Tensors {
        torch::zeros({1, nm, nm, num_terms}, options),
        torch::zeros({num_terms, ng, ng, 1}, options)});

      // Even spherical harmonic terms
      for (int64_t m = 0; m < (l + 1); m++) {
        const double m_f = m;

        // Compute the normalization factor
        double log_front = std::log(two * l_f + one);
        double log_factor =
          std::lgamma(l_f - m_f + one) - std::lgamma(l_f + m_f + one);
        double scale =
          (m != 0 ? two : one) *
          std::exp(static_cast<double>(0.5) * (log_front + log_factor));

        // Compute associated Legendre polynomials and the normalization
        // factor
        auto plm = scale * math::special::assoc_legendre(l, m, mu, false);
        Yl[0].index_put_({0, slice, slice, m}, torch::outer(plm, plm));

        // Compute the cosine factor
        auto cos_gamma = torch::cos(m_f * gamma);
        Yl[1].index_put_(
          {m, slice, slice, 0}, torch::outer(cos_gamma, cos_gamma));
      }

      // Odd spherical harmonic terms
      for (int64_t m = l + 1; m < num_terms; m++) {
        // Copy associated Legendre polynomials and the normalization factor
        Yl[0].index_put_(
          {0, slice, slice, m}, Yl[0].index({0, slice, slice, m - l}));

        // Compute the sine factor
        auto sin_gamma = torch::sin(static_cast<double>(m - l) * gamma);
        Yl[1].index_put_(
          {m, slice, slice, 0}, torch::outer(sin_gamma, sin_gamma));
      }

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
std::tuple<linalg::Operator, linalg::Operator> DGFirstOrderTransportBackend<
  cad::Patch, Fmt, NumDim>::assemble_boundary_operators(size_t dim,
  bool is_upper)
{
  // Return nothing if the boundary is degenerate
  if (block_->get_boundary_info(dim, is_upper).get_type() ==
      BoundaryType::DEGENERATE) {
    return std::make_tuple(linalg::Operator(), linalg::Operator());
  }

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

    // Get the outflow and inflow boundary operators
    auto B_out = assemble_outflow_boundary_operator(basis, normal, mapping);
    auto B_in = assemble_inflow_boundary_operator(basis, normal, mapping,
      block_->get_boundary_info(dim, is_upper).get_type());

    // Create a core for the new dimension with a 1.0 in the spot of
    // the interpolatory control point evaluation
    size_t target_idx = 2 + dim;
    int64_t num_ctrlpts = block_->get_ctrlpts_size(dim);
    auto target_core = torch::zeros({num_ctrlpts, num_ctrlpts}, options);
    if (is_upper) {
      target_core[-1][-1] = 1.0;
    } else {
      target_core[0][0] = 1.0;
    }
    target_core.unsqueeze_(0).unsqueeze_(-1);

    // Create an identity core for energy
    auto energy = torch::eye(material_->get_num_groups(), options)
                    .unsqueeze_(0)
                    .unsqueeze_(-1);

    auto inject_basis_and_energy = [&](const linalg::TTEngine& B) {
      // Get the cores
      auto B_cores = B.get_cores();
      B_cores.reserve(B_cores.size() + 2);

      // Inject the core either in between ranks or at the end of the TT
      if (target_idx < NumDim + 1) {
        int64_t r = B_cores[target_idx].size(0);
        B_cores.insert(B_cores.begin() + target_idx,
          torch::eye(r, options).reshape({r, 1, 1, r}) * target_core);

      } else {
        B_cores.push_back(target_core);
      }

      // Append an energy core
      B_cores.push_back(energy);

      return linalg::TTEngine(std::move(B_cores), false);
    };

    return std::make_tuple(
      linalg::Operator(
        inject_basis_and_energy(std::move(B_out)).diagonalize({0, 1})),
      B_in.has_value()
        ? linalg::Operator(inject_basis_and_energy(std::move(*B_in)))
        : linalg::Operator());

  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    const auto& options =
      torch::TensorOptions().device(block_->get_device()).dtype(config_->dtype);
    double orientation = block_->get_orientation() * (is_upper ? 1.0 : -1.0);

    linalg::TTEngine normal(
      {torch::tensor({orientation}, options).reshape({1, 1, 1, 1})});

    auto B_out = assemble_outflow_boundary_operator({}, {normal}, {});
    auto B_in = assemble_inflow_boundary_operator(
      {}, {normal}, {}, block_->get_boundary_info(dim, is_upper).get_type());

    // Create spatial core
    int64_t num_ctrlpts = block_->get_ctrlpts_size(dim);
    auto target_core = torch::zeros({num_ctrlpts, num_ctrlpts}, options);
    if (is_upper) {
      target_core[-1][-1] = 1.0;
    } else {
      target_core[0][0] = 1.0;
    }
    target_core.unsqueeze_(0).unsqueeze_(-1);

    // Create identity core for energy groups
    auto energy = torch::eye(material_->get_num_groups(), options)
                    .unsqueeze_(0)
                    .unsqueeze_(-1);

    auto inject_basis_and_energy = [&](const linalg::TTEngine& B) {
      auto B_cores = B.get_cores();
      B_cores.reserve(B_cores.size() + 2);

      // Append space and energy cores
      B_cores.push_back(target_core);
      B_cores.push_back(energy);

      return linalg::TTEngine(std::move(B_cores), false);
    };

    return std::make_tuple(
      linalg::Operator(
        inject_basis_and_energy(std::move(B_out)).diagonalize({0})),
      B_in.has_value()
        ? linalg::Operator(inject_basis_and_energy(std::move(*B_in)))
        : linalg::Operator());
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

    if (condition == BoundaryType::INTERNAL) {
      return assemble_interface_boundary_operator(
        basis, normal, mapping, false);
    }
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_outflow_boundary_operator",
    "This method does not support this format yet");
}

template<FormatType Fmt, int64_t NumDim>
typename Return<Fmt, NumDim>::Type DGFirstOrderTransportBackend<cad::Patch, Fmt,
  NumDim>::assemble_interface_boundary_operator(const ReturnType& basis,
  const typename Return<Fmt, NumDim>::VectorType& normal,
  const ReturnType& mapping, bool is_outflow)
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
    linalg::TTEngine B =
      static_cast<double>(0.5) *
      (ando + (is_outflow ? ndo : -ndo))
        .round(config_->rounding.eps, config_->rounding.max_rank);

    // Apply the basis
    for (size_t i = 0; i < NumDim - 1; i++) {
      auto& B_core = B[i + 2];
      int64_t rl_b = basis[i].size(0);
      int64_t m = basis[i].size(1);
      int64_t n = basis[i].size(2);

      B_core = torch::einsum("abcd,ebfg->aebfdg", {B_core, basis[i]})
                 .reshape({B_core.size(0) * rl_b, m, n, -1});
    }
    B.round_(config_->rounding.eps, config_->rounding.max_rank);

    // Compute the outer product with the mapped basis
    for (size_t i = 0; i < NumDim - 1; i++) {
      auto& B_core = B[i + 2];
      int64_t rl_b = basis[i].size(0);
      int64_t m = B_core.size(2);
      int64_t n = basis[i].size(2);

      auto mapped_basis = basis[i] * mapping[i];

      B_core = torch::einsum("abcd,ebfg->aecfdg", {B_core, mapped_basis})
                 .reshape({B_core.size(0) * rl_b, m, n, -1});
    }
    B.round_(config_->rounding.eps, config_->rounding.max_rank);

    return B;
  } else if constexpr (Fmt == FormatType::TENSOR_TRAIN && NumDim == 1) {
    // Get ordinates
    torch::Tensor ordinates = assemble_ordinates()[0][0];

    // Compute dot(ordinates, normal)
    torch::Tensor ndo = ordinates * normal[0][0];

    // Get either the inflow or outflow angular component of the boundary
    // operator
    return linalg::TTEngine(
      {torch::clamp(is_outflow ? ndo : -ndo, 0).reshape({1, -1, 1, 1})}, false);
  }

  throw utils::runtime_error("ttnte::physics::DIGAFirstOrderTransportBackend::"
                             "assemble_interface_boundary_operator",
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
