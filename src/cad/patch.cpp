#include "ttnte/cad/patch.hpp"
#include "ttnte/cad/bspline_basis.hpp"
#include "ttnte/cad/patch_kernels.hpp"
#include "ttnte/linalg/binomial.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/io_formatting.hpp"
#include <sstream>

namespace {

void check_parametric_coords(const ttnte::cad::Patch& patch,
  const torch::Tensor& u, int64_t expected_ndim,
  const torch::Device& expected_device, const torch::ScalarType& expected_dtype,
  const std::string& func_name)
{
  if (!u.defined() || u.ndimension() != expected_ndim || u.numel() == 0) {
    throw ttnte::utils::runtime_error(patch, func_name,
      "The parametric coordinates tensor must be a defined " +
        std::to_string(expected_ndim) + "-dimensional tensor");
  }
  if (u.device() != expected_device || u.dtype() != expected_dtype) {
    throw ttnte::utils::runtime_error(patch, func_name,
      "All parametric coordinate tensors must have the same dtype and "
      "device");
  }
  if (torch::logical_or(u < 0.0, u > 1.0).any().item<bool>()) {
    throw ttnte::utils::runtime_error(
      patch, func_name, "All parametric coordinates should be between 0 and 1");
  }
}

} // namespace

namespace ttnte::cad {

// =================================================================
// Private constructors
Patch::Patch(std::optional<std::string> label) : Base(label) {}

Patch::Patch(const torch::Tensor& ctrlpts, const Basis& basis, bool is_rational,
  std::optional<std::string> label)
  : Base(label)
{
  set_basis(basis);
  set_ctrlptsw(ctrlpts);
  is_rational_ = is_rational;
}

Patch::Patch(const torch::Tensor& ctrlpts, const torch::Tensor& weights,
  const Basis& basis, std::optional<std::string> label)
  : Base(label)
{
  set_basis(basis);
  set_ctrlpts(ctrlpts);
  set_weights(weights);
}

// =================================================================
// Public methods
void Patch::finalize_impl()
{
  if (basis_.empty() || !ctrlptsw_.defined()) {
    throw utils::runtime_error(*this, error_context("finalize_impl"),
      "Both the basis and control points must be set before validation");
  }

  // Check the dimensionality
  if (basis_.size() != ctrlptsw_.ndimension() - 1) {
    std::stringstream ss;
    ss << basis_.size() << "-dimensional basis expects a " << basis_.size() + 1
       << "-dimensional control point tensor\n"
       << "A " << ctrlptsw_.ndimension()
       << "-dimensional control point tensor was given";

    throw utils::runtime_error(*this, error_context("finalize_impl"), ss.str());
  }

  // Check the control point tensor shape matches the basis
  for (size_t i = 0; i < basis_.size(); i++) {
    if (basis_[i].get_size() != ctrlptsw_.size(i)) {
      throw utils::runtime_error(*this, error_context("finalize_impl"),
        "There must be an equal number of basis functions and control points"
        "along each dimension");
    }
  }

  // Check device and data type
  auto device = ctrlptsw_.device();
  auto dtype = ctrlptsw_.dtype();
  for (const auto& b : basis_) {
    if (b.get_device() != device || b.get_dtype() != dtype) {
      throw utils::runtime_error(*this, error_context("finalize_impl"),
        "The basis and control points should be on the same device with the "
        "same data type");
    }
  }

  // Combine all data into one tensors
  c10::SmallVector<torch::Tensor, 4> tensors;
  tensors.reserve(basis_.size());

  for (const auto& b : basis_) {
    tensors.push_back(b.get_knotvector());
  }
  tensors.push_back(ctrlptsw_.flatten());

  // Make one large tensor for MPI
  meshdata_ = torch::cat(tensors);

  // Create views for the knot vector and control points
  int64_t idx = 0;
  for (auto& b : basis_) {
    // Length of the knot vector
    int64_t kv_size = b.get_knotvector().size(0);

    // Finalize basis and increment index
    b.finalize(meshdata_.narrow(0, idx, kv_size));
    idx += kv_size;
  }

  // Set the control point view
  int64_t cpw_size = ctrlptsw_.numel();
  auto cpw_shape = ctrlptsw_.sizes();
  ctrlptsw_ = meshdata_.narrow(0, idx, cpw_size).view(cpw_shape);

  is_finalized_ = true;

  // Compute the Jacobian at the centroid of the patch to determine the
  // orientation of the patch
  int64_t ndim = get_ndim();
  int64_t phys_ndim =
    is_rational_ ? ctrlptsw_.size(-1) - 1 : ctrlptsw_.size(-1);
  auto jac = evaluate_all_jacobian(c10::SmallVector<torch::Tensor, 3>(ndim,
                                     torch::tensor({0.5}, ctrlptsw_.options())))
               .reshape({phys_ndim, ndim});

  // Compute the determinant and the condition number
  double cond = torch::linalg_cond(jac).item<double>();

  // Check for degeneracies
  double max_cond = jac.scalar_type() == torch::kFloat64 ? 1e12 : 1e5;
  if (std::isnan(cond) || std::isinf(cond) || cond > max_cond) {
    throw utils::runtime_error(*this, error_context("finalize_impl"),
      "Patch is degenerate, or pinched at its center point. Condition number\n"
      "of " +
        std::to_string(cond) + "exceeds precision limits.");
  }

  if (phys_ndim == ndim) {
    double det = torch::linalg_det(jac).item<double>();
    orientation_ = (det >= 0.0) ? 1.0 : -1.0;
  } else if (phys_ndim < ndim) {
    throw utils::runtime_error(*this, error_context("finalize_impl"),
      "Patches cannot have more parametric dimensions than physical");
  }
}

torch::Tensor Patch::evaluate(
  const c10::SmallVector<torch::Tensor, 3>& local_coords) const
{
  is_finalized_or_error("evaluate");

  // Expected tensor options
  const auto& options = local_coords[0].options();

  // Check input
  if (local_coords.size() != get_ndim()) {
    throw utils::runtime_error(*this, error_context("evaluate"),
      "The length of `local_coords` must be equal to the dimension of the "
      "spline");
  }
  for (const auto& u : local_coords) {
    // Check these are each 1-D
    if (u.ndimension() != 1 || !u.defined()) {
      throw utils::runtime_error(*this, error_context("evaluate"),
        "Each axis of points must be 1-dimensional and defined");
    } else if (u.device() != options.device() || u.dtype() != options.dtype()) {
      throw utils::runtime_error(*this, error_context("evaluate"),
        "All tensors given in `local_coords` must data type and on the same "
        "device");
    }
  }

  // Apply one basis at a time
  int64_t phys_d = ctrlptsw_.size(-1);
  auto phys_coords = ctrlptsw_.to(options);

  for (size_t d = 0; d < basis_.size(); d++) {
    const auto& b = basis_[d];
    const auto& p = b.get_degree();
    const auto& u = local_coords[d];

    // Find spans for each coordinate
    auto spans = b.find_spans(u);

    // Gather indices: spans - p + j
    auto j_range = torch::arange(p + 1, spans.options()).unsqueeze(0);
    auto idx = spans.unsqueeze(1) - p + j_range;

    // Move the dimension of interest to the front
    phys_coords = torch::movedim(phys_coords, d, 0);

    // Get the current shape
    auto shape = phys_coords.sizes().vec();
    shape[0] = u.size(0);

    // Flatten the tensor
    auto phys_coords_flat = phys_coords.reshape({phys_coords.size(0), -1});

    // Compute contraction
    phys_coords = torch::matmul(b.evaluate(u, spans).unsqueeze(1),
      torch::embedding(phys_coords_flat, idx));

    // Reshape back and permute dim back
    phys_coords = torch::movedim(phys_coords.reshape(shape), 0, d);
  }

  return is_rational_ ? phys_coords.slice(-1, 0, -1) / phys_coords.slice(-1, -1)
                      : phys_coords;
}

torch::Tensor Patch::evaluate(
  const torch::Tensor& local_coords, int64_t derivative_order) const
{
  is_finalized_or_error("evaluate");

  if (local_coords.ndimension() != 2) {
    throw utils::runtime_error(
      *this, error_context("evaluate"), "`local_coords` must be 2-dimensional");
  } else if (local_coords.size(1) != basis_.size()) {
    throw utils::runtime_error(*this, error_context("evaluate"),
      "The second dimension of `local_coords` should be size " +
        std::to_string(basis_.size()));
  }
  if (derivative_order < 0) {
    throw utils::runtime_error(*this, error_context("evaluate"),
      "`derivative_order` must be greater than or equal to 0");
  }

  int64_t n = local_coords.size(0);
  int64_t ndim = get_ndim();

  // Contract ctrlptsw_ against, for each dimension, a single requested
  // derivative order (0 unless overridden in `dim_orders`). Returns the raw
  // (undivided, weight channel included if rational) numerator.
  auto contract = [&](const c10::SmallVector<int64_t, 3>& dim_orders) {
    auto phys_coords = ctrlptsw_.to(local_coords.options());

    for (size_t d = 0; d < basis_.size(); d++) {
      const auto& b = basis_[d];
      const auto& p = b.get_degree();
      const auto& u = local_coords.select(1, d);

      auto spans = b.find_spans(u);
      auto j_range = torch::arange(p + 1, spans.options()).unsqueeze(0);
      auto idx = spans.unsqueeze(1) - p + j_range;

      // Basis values for exactly the requested order at this dimension.
      auto order = dim_orders[d];
      auto evals = b.evaluate(u, spans, order);
      auto weight = order == 0 ? evals : evals.select(1, order);

      if (d == 0) {
        auto shape = phys_coords.sizes().vec();
        shape[0] = -1;

        auto phys_coords_flat = phys_coords.view({phys_coords.size(0), -1});
        phys_coords = torch::matmul(
          weight.unsqueeze(1), torch::embedding(phys_coords_flat, idx));
        phys_coords = phys_coords.view(shape);

      } else {
        auto shape = phys_coords.sizes().vec();
        shape.erase(shape.begin() + 1);

        int64_t n_b = phys_coords.size(1);
        auto phys_coords_flat = phys_coords.view({n, n_b, -1});

        phys_coords = torch::matmul(weight.unsqueeze(1),
          torch::gather(phys_coords_flat, /*dim=*/1,
            idx.unsqueeze(2).expand({n, p + 1, phys_coords_flat.size(-1)})));
        phys_coords = phys_coords.view(shape);
      }
    }

    return phys_coords; // (n, phys_d [+1 if rational]), not yet divided
  };

  c10::SmallVector<int64_t, 3> zero_orders(ndim, 0);

  if (derivative_order == 0) {
    auto A = contract(zero_orders);
    return is_rational_ ? A.slice(1, 0, -1) / A.slice(1, -1) : A;
  }

  // Per-dimension derivative order actually used, clamped to each
  // dimension's degree -- matches evaluate_all_basis()'s convention.
  c10::SmallVector<int64_t, 3> orders(ndim);
  for (int64_t d = 0; d < ndim; d++) {
    orders[d] = std::min(derivative_order, basis_[d].get_degree());
  }

  // Slot 0 is the value; slots thereafter are, in turn, each dimension's
  // 1st..orders[d]-th derivative (all other dimensions at order 0).
  c10::SmallVector<torch::Tensor, 6> slots;
  slots.push_back(contract(zero_orders));

  for (int64_t d = 0; d < ndim; d++) {
    for (int64_t k = 1; k <= orders[d]; k++) {
      auto dim_orders = zero_orders;
      dim_orders[d] = k;
      slots.push_back(contract(dim_orders));
    }
  }
  auto stacked = torch::stack(slots, /*dim=*/-1); // (n, phys_d[+1], slots)

  if (!is_rational_) {
    return stacked;
  }

  // NURBS quotient-rule correction: R^(k) = (A^(k) - sum_{i=1}^{k}
  // binom(k,i) * W^(i) * R^(k-i)) / W^(0), applied per physical channel
  // (the weight channel is included too -- it trivially corrects to a
  // constant 1 with all derivatives 0, and is stripped from the return).
  auto W = stacked.select(1, -1); // (n, slots)
  auto W0 = W.select(-1, 0);      // (n,)

  auto result = stacked.clone();
  result.select(-1, 0).div_(W0.unsqueeze(-1));

  int64_t pos = 1;
  for (int64_t d = 0; d < ndim; d++) {
    for (int64_t k = 1; k <= orders[d]; k++) {
      int64_t k_idx = pos + k - 1;
      auto Ak = stacked.select(-1, k_idx);
      auto sum_term = torch::zeros_like(Ak);

      for (int64_t i = 1; i <= k; i++) {
        int64_t binom = linalg::binomial.get(k, i);
        int64_t i_idx = pos + i - 1;
        int64_t k_minus_i_idx = (k == i) ? 0 : pos + (k - i) - 1;

        auto Wi = W.select(-1, i_idx);
        auto R_k_minus_i = result.select(-1, k_minus_i_idx);
        sum_term.addcmul_(Wi.unsqueeze(-1), R_k_minus_i, binom);
      }

      result.select(-1, k_idx).copy_((Ak - sum_term) / W0.unsqueeze(-1));
    }
    pos += orders[d];
  }

  return result.slice(1, 0, -1);
}

torch::Tensor Patch::evaluate_jacobian(const torch::Tensor& local_coords) const
{
  int64_t ndim = get_ndim();
  auto full = evaluate(local_coords, 1); // (n, phys_d, 1 + ndim)
  return full.slice(-1, 1, 1 + ndim);
}

InverseMapResult Patch::inverse_map(const torch::Tensor& physical_coords,
  int64_t max_iter, double tol, std::optional<torch::Tensor> initial_guess,
  int64_t seed_resolution) const
{
  is_finalized_or_error("inverse_map");

  if (physical_coords.ndimension() != 2) {
    throw utils::runtime_error(*this, error_context("inverse_map"),
      "`physical_coords` must be 2-dimensional");
  }

  int64_t ndim = get_ndim();
  int64_t n = physical_coords.size(0);
  auto options = physical_coords.options();

  torch::Tensor u;
  if (initial_guess.has_value()) {
    u = initial_guess.value().clone();

  } else {
    // Coarse-grid closest-point search for a starting guess, mirroring the
    // legacy ttnte/iga/mesh.py inverse_map()'s coarse-mesh seeding (but at
    // `seed_resolution` instead of that legacy code's fixed low resolution
    // -- see DEFAULT_INVERSE_MAP_SEED_RESOLUTION's comment). Like that
    // legacy code, the seed grid deliberately EXCLUDES the parametric
    // boundary (u/v == 0 or 1): a patch with a collapsed edge (e.g. a
    // ruled surface off a degenerate point, as used for a pie-slice/wedge
    // patch) has an exactly-singular Jacobian all along that edge, so
    // seeding a target there is a trap, not just a slow start -- the
    // Newton-Raphson step below can never correct the coordinate along the
    // collapsed direction once clamped onto the boundary (that direction's
    // Jacobian column is exactly zero), and any step off the domain just
    // gets re-clamped back onto it every iteration.
    c10::SmallVector<torch::Tensor, 3> grid_axes(ndim);
    for (int64_t d = 0; d < ndim; d++) {
      grid_axes[d] =
        torch::linspace(0, 1, seed_resolution + 2, options)
          .slice(/*dim=*/0, /*start=*/1, /*end=*/seed_resolution + 1);
    }

    auto grid_phys = evaluate(grid_axes); // (res, ..., res, phys_d)
    int64_t phys_d = grid_phys.size(-1);
    auto grid_phys_flat = grid_phys.reshape({-1, phys_d}); // (res^ndim, phys_d)

    std::vector<torch::Tensor> mesh = torch::meshgrid(
      std::vector<torch::Tensor>(grid_axes.begin(), grid_axes.end()), "ij");
    std::vector<torch::Tensor> mesh_flat;
    mesh_flat.reserve(ndim);
    for (auto& m : mesh) {
      mesh_flat.push_back(m.reshape({-1}));
    }
    auto grid_params = torch::stack(mesh_flat, -1); // (res^ndim, ndim)

    // (n, res^ndim): squared distance from each target to each grid point.
    auto dists = (grid_phys_flat.unsqueeze(0) - physical_coords.unsqueeze(1))
                   .pow(2)
                   .sum(-1);
    auto closest = dists.argmin(-1); // (n,)
    u = grid_params.index_select(0, closest);
  }

  torch::Tensor residual;
  torch::Tensor converged;

  for (int64_t iter = 0; iter < max_iter; iter++) {
    auto F = evaluate(u) - physical_coords; // (n, phys_d)
    residual = F.norm(2, -1);               // (n,)
    converged = residual < tol;

    if (converged.all().item<bool>()) {
      break;
    }

    // Least-squares Newton step: delta = argmin_d || J @ d + F ||. Solving
    // via lstsq (QR with pivoting on CPU) rather than forming the normal
    // equations avoids squaring the condition number and degrades
    // gracefully (rather than throwing) for rank-deficient Jacobians, e.g.
    // when a candidate point sits far outside this patch's valid domain.
    auto J = evaluate_jacobian(u); // (n, phys_d, ndim)
    auto delta =
      -std::get<0>(torch::linalg_lstsq(J, F.unsqueeze(-1),
                     /*rcond=*/std::nullopt, /*driver=*/std::nullopt))
         .squeeze(-1); // (n, ndim)

    auto mask = torch::logical_not(converged).unsqueeze(-1);
    u = torch::where(mask, (u + delta).clamp(0.0, 1.0), u);
  }

  return InverseMapResult {
    std::move(u), std::move(residual), std::move(converged)};
}

torch::Tensor Patch::evaluate_field(
  const torch::Tensor& field, const torch::Tensor& points) const
{
  is_finalized_or_error("evaluate_field");

  if (points.ndimension() != 2) {
    throw utils::runtime_error(
      *this, error_context("evaluate_field"), "`points` must be 2-dimensional");
  } else if (points.size(1) != static_cast<int64_t>(basis_.size())) {
    throw utils::runtime_error(*this, error_context("evaluate_field"),
      "The second dimension of `points` should be size " +
        std::to_string(basis_.size()));
  }

  int64_t n = points.size(0);

  // Same per-dimension basis contraction evaluate(Tensor, 0) uses on
  // ctrlptsw_, applied directly to an arbitrary `coeffs` tensor instead --
  // this is purely a function of this patch's own (already-finalized,
  // read-only) basis, with no geometric validity constraints of its own.
  auto contract = [&](const torch::Tensor& coeffs) -> torch::Tensor {
    auto values = coeffs.to(points.options());

    for (size_t d = 0; d < basis_.size(); d++) {
      const auto& b = basis_[d];
      const auto& p = b.get_degree();
      const auto& u = points.select(1, d);

      auto spans = b.find_spans(u);
      auto j_range = torch::arange(p + 1, spans.options()).unsqueeze(0);
      auto idx = spans.unsqueeze(1) - p + j_range;
      auto weight = b.evaluate(u, spans);

      if (d == 0) {
        auto shape = values.sizes().vec();
        shape[0] = -1;

        auto values_flat = values.view({values.size(0), -1});
        values = torch::matmul(
          weight.unsqueeze(1), torch::embedding(values_flat, idx));
        values = values.view(shape);

      } else {
        auto shape = values.sizes().vec();
        shape.erase(shape.begin() + 1);

        int64_t n_b = values.size(1);
        auto values_flat = values.view({n, n_b, -1});

        values = torch::matmul(weight.unsqueeze(1),
          torch::gather(values_flat, /*dim=*/1,
            idx.unsqueeze(2).expand({n, p + 1, values_flat.size(-1)})));
        values = values.view(shape);
      }
    }

    return values; // (n, num_channels)
  };

  if (!is_rational_) {
    return contract(field);
  }

  // NURBS quotient rule: `field`'s DOF coefficients are coefficients of the
  // RATIONAL basis R_i(u) = N_i(u) * w_i / sum_j N_j(u) * w_j (the same
  // basis assemble_basis() builds and the solve operates against) -- so
  // sum_i R_i(u) * field_i is recovered by contracting the plain B-spline
  // basis against the WEIGHTED field, divided by the weight function
  // contracted the same way (mirrors how ctrlptsw_ stores geometry
  // pre-weighted, so evaluate()'s own quotient rule needs no separate
  // pre-weighting step -- field, unlike ctrlptsw_, arrives unweighted).
  torch::Tensor weights = get_weights();
  torch::Tensor weighted_field = field * weights.unsqueeze(-1);
  torch::Tensor numerator = contract(weighted_field);
  torch::Tensor denominator = contract(weights.unsqueeze(-1));
  return numerator / denominator;
}

std::string Patch::to_string_impl() const
{
  std::stringstream ss;
  ss << "Patch(ctrlpts_size=";

  if (ctrlptsw_.defined()) {
    std::string sep = "(";
    for (const auto& n : ctrlptsw_.sizes()) {
      ss << sep << n;
      sep = ", ";
    }
    ss << "), is_rational=" << (is_rational_ ? "True" : "False")
       << ", label=" << label_ << ")";

  } else {
    ss << "(), is_rational=False, label=" << label_ << ")";
  }

  return ss.str();
}

torch::Tensor Patch::evaluate_all_basis(
  const c10::SmallVector<torch::Tensor, 3>& local_coords,
  int64_t derivative_order, bool eval_cross_terms) const
{
  is_finalized_or_error("evaluate_basis");
  const auto& options = local_coords[0].options();
  const int64_t ndim = get_ndim();

  if (local_coords.size() != ndim) {
    throw utils::runtime_error(*this, error_context("evaluate_basis"),
      "The length of `local_coords` must equal the patch dimension");
  }
  if (derivative_order < 0) {
    throw utils::runtime_error(*this, error_context("evaluate_basis"),
      "`derivative_order` must be greater than or equal to 0");
  }

  c10::SmallVector<int64_t, 3> orders;
  orders.reserve(ndim);
  for (size_t d = 0; d < ndim; d++) {
    // Run checks on the given parametric coordinates
    check_parametric_coords(*this, local_coords[d], 1, get_device(),
      get_dtype(), error_context("evaluate_all_basis"));
    orders.push_back(std::min(derivative_order, basis_[d].get_degree()));
  }

  // Broadcasting helper function
  auto get_broadcast_shape = [&](size_t d) {
    if (eval_cross_terms) {
      c10::SmallVector<int64_t, 9> shape(3 * ndim, 1);
      shape[d] = local_coords[d].size(0);
      shape[ndim + d] = orders[d] + 1;
      shape[2 * ndim + d] = basis_[d].get_size();
      return shape;
    } else {
      c10::SmallVector<int64_t, 9> shape(2 * ndim + 1, 1);
      shape[d] = local_coords[d].size(0);
      shape[ndim] = orders[d] + 1;
      shape[ndim + 1 + d] = basis_[d].get_size();
      return shape;
    }
  };

  // Compute the B-spline basis functions and their derivatives
  torch::Tensor result;
  if (!eval_cross_terms && derivative_order > 0) {
    // Allocate the result tensor
    c10::SmallVector<int64_t, 9> shape(2 * ndim + 1, 1);
    for (size_t d = 0; d < ndim; d++) {
      const auto& b = basis_[d];

      shape[d] = local_coords[d].size(0); // Number of points along dimension d
      shape[ndim] += orders[d];           // Total number of derivatives
      shape[ndim + 1 + d] =
        b.get_size(); // Number of control points along dimension d
    }
    result = torch::empty(shape, options);

    // Compute the derivatives and populate them accordingly
    int64_t pos = 1;
    for (size_t d = 0; d < ndim; d++) {
      const auto& b = basis_[d];
      const auto& u = local_coords[d];
      int64_t order = orders[d];

      auto evals =
        b.evaluate_all(u, derivative_order).reshape(get_broadcast_shape(d));
      auto der0 = evals.narrow(ndim, 0, 1);

      if (d == 0) {
        result.copy_(der0);
        if (order > 0) {
          result.narrow(ndim, 1, order).copy_(evals.slice(ndim, 1));
        }

      } else {
        // Multiply the zeroth derivative up to just before the derivatives of
        // this dimension
        result.narrow(ndim, 0, pos).mul_(der0);

        // Multiply the zeroth derivative for all entries past the derivatives
        // of this dimension
        if (pos + order < shape[ndim]) {
          result.slice(ndim, pos + order).mul_(der0);
        }

        if (order > 0) {
          result.narrow(ndim, pos, order).mul_(evals.slice(ndim, 1));
        }
      }
      pos += order;
    }
  } else {
    // Compute all cross term derivatives for a B-spline
    for (size_t d = 0; d < ndim; d++) {
      const auto& b = basis_[d];
      const auto& u = local_coords[d];

      // Get the broadcasting shape
      auto shape = get_broadcast_shape(d);

      result = (d == 0)
                 ? b.evaluate_all(u, derivative_order).reshape(shape)
                 : result * b.evaluate_all(u, derivative_order).reshape(shape);
    }
  }

  // Return B-spline result
  if (!is_rational_) {
    if (derivative_order == 0) {
      auto shape = result.sizes().vec();
      shape.erase(shape.begin() + ndim, shape.end() - ndim);
      return result.reshape(shape);
    }
    return result;
  }

  // Apply the weight tensor
  c10::SmallVector<int64_t, 9> w_shape(
    eval_cross_terms ? (3 * ndim) : (2 * ndim + 1), 1);
  c10::SmallVector<int64_t, 3> sum_dims;
  for (size_t i = eval_cross_terms ? (2 * ndim) : (ndim + 1);
    i < w_shape.size(); i++) {
    w_shape[i] = result.size(i);
    sum_dims.push_back(i);
  }
  result.mul_(get_weights().reshape(w_shape));

  // Compute the sum over all control points
  auto W = result.sum(sum_dims, true);

  // Return early if we want no derivatives
  if (derivative_order == 0) {
    result.div_(W);
    auto shape = result.sizes().vec();
    shape.erase(shape.begin() + ndim, shape.end() - ndim);
    return result.reshape(shape);
  }

  if (!eval_cross_terms) {
    // Compute the rational zeroth derivative
    auto W0 = W.select(ndim, 0);
    result.select(ndim, 0).div_(W0);

    // Compute pure rational derivatives incrementally
    int64_t pos = 1;
    for (size_t d = 0; d < ndim; d++) {
      int64_t order = orders[d];

      for (int64_t k = 1; k <= order; k++) {
        int64_t k_idx = pos + k - 1;

        // A^(k) currently lives in result.select(ndim, k_idx)
        auto Rk = result.select(ndim, k_idx);
        torch::Tensor sum_term = torch::zeros_like(Rk);

        // Summation: binom(k, i) * W^(i) * R^(k-i)
        for (int64_t i = 1; i <= k; i++) {
          int64_t binom = linalg::binomial.get(k, i);
          int64_t i_idx = pos + i - 1;
          int64_t k_minus_i_idx = (k == i) ? 0 : pos + (k - i) - 1;

          auto Wi = W.select(ndim, i_idx);
          auto R_k_minus_i = result.select(ndim, k_minus_i_idx);

          // sum_term += binom * Wi * R_k_minus_i
          sum_term.addcmul_(Wi, R_k_minus_i, binom);
        }

        // R^(k) = (A^(k) - sum_term) / W^(0)
        // Since Rk is a view into result, this updates the tensor in-place
        Rk.sub_(sum_term).div_(W0);
      }
      pos += order;
    }
    return result;

  } else {
    // Odometer indexing for multi-index
    c10::SmallVector<int64_t, 3> k_idx(ndim, 0);
    c10::SmallVector<int64_t, 3> k_lim;
    k_lim.reserve(ndim);
    for (size_t i = ndim; i < 2 * ndim; i++) {
      k_lim.push_back(result.size(i) - 1);
    }
    auto increment_index = [](c10::SmallVector<int64_t, 3>& multi_index,
                             const c10::SmallVector<int64_t, 3>& limits) {
      for (int i = multi_index.size() - 1; i >= 0; --i) {
        if (++multi_index[i] <= limits[i]) {
          return true;
        }
        multi_index[i] = 0;
      }
      return false;
    };
    auto get_linear_index = [&](
                              const c10::SmallVector<int64_t, 3>& multi_index) {
      int64_t linear_idx = 0;
      int64_t stride = 1;
      for (int i = ndim - 1; i >= 0; --i) {
        linear_idx += multi_index[i] * stride;
        stride *= k_lim[i] + 1;
      }
      return linear_idx;
    };

    // Flatten both tensors
    auto final_shape = result.sizes();
    result = result.flatten(ndim, 2 * ndim - 1);
    W = W.flatten(ndim, 2 * ndim - 1);

    // Base weight tensor and zeroth derivative
    auto W0 = W.narrow(ndim, 0, 1);
    result.narrow(ndim, 0, 1).div_(W0);

    // Recursive loop
    while (increment_index(k_idx, k_lim)) {
      int64_t k_linear = get_linear_index(k_idx);

      // Get numerator term
      torch::Tensor Ak = result.narrow(ndim, k_linear, 1);

      // Sum over contributing slices
      torch::Tensor sum_term = torch::zeros_like(Ak);
      c10::SmallVector<int64_t, 3> i_idx(ndim, 0);

      while (increment_index(i_idx, k_idx)) {
        // Multi-index Binomial coefficient
        int64_t binom = 1;
        c10::SmallVector<int64_t, 3> k_minus_i(ndim);

        for (size_t d = 0; d < ndim; d++) {
          binom *= linalg::binomial.get(k_idx[d], i_idx[d]);
          k_minus_i[d] = k_idx[d] - i_idx[d];
        }

        // Index out relevant tensors
        torch::Tensor Wi = W.narrow(ndim, get_linear_index(i_idx), 1);
        torch::Tensor prev =
          result.narrow(ndim, get_linear_index(k_minus_i), 1);

        // sum_term += binom * Wi * prev
        sum_term.addcmul_(Wi, prev, binom);
      }

      // Place the result back into the results tensor
      result.narrow(ndim, k_linear, 1).copy_((Ak - sum_term) / W0);
    }

    return result.reshape(final_shape);
  }
}

torch::Tensor Patch::evaluate_all_jacobian(
  const c10::SmallVector<torch::Tensor, 3>& local_coords) const
{
  is_finalized_or_error("evaluate_all_jacobian");
  int64_t ndim = get_ndim();

  // Evaluate the basis and the first derivatives
  auto evals = evaluate_all_basis(local_coords, 1, false).narrow(ndim, 1, ndim);

  // Get the unweighted control points
  torch::Tensor ctrlpts = get_ctrlpts();

  // Get the dimensions to contract over
  c10::SmallVector<int64_t, 3> dims_evals(ndim);
  c10::SmallVector<int64_t, 3> dims_ctrlpts(ndim);
  for (int64_t i = 0; i < ndim; i++) {
    dims_evals[i] = ndim + 1 + i;
    dims_ctrlpts[i] = i;
  }

  return torch::tensordot(evals, ctrlpts, dims_evals, dims_ctrlpts)
    .transpose(-2, -1);
}

torch::Tensor Patch::evaluate_basis(
  const c10::SmallVector<torch::Tensor, 3>& local_coords,
  const int64_t& derivative_order) const
{
  is_finalized_or_error("evaluate_basis");

  if (local_coords.size() != get_ndim()) {
    throw utils::runtime_error(*this, error_context("evaluate_basis"),
      "The length of `local_coords` must equal the patch dimension");
  }
  if (derivative_order < 0) {
    throw utils::runtime_error(*this, error_context("evaluate_basis"),
      "`derivative_order` must be greater than or equal to 0");
  }

  const auto& options = local_coords[0].options();
  const int64_t ndim = get_ndim();

  c10::SmallVector<torch::Tensor, 3> basis_per_dim;
  basis_per_dim.reserve(ndim);

  c10::SmallVector<int64_t, 6> interleaved_shape;
  interleaved_shape.reserve(2 * ndim);

  for (size_t d = 0; d < local_coords.size(); ++d) {
    const auto& u = local_coords[d];

    if (!u.defined() || u.ndimension() != 1) {
      throw utils::runtime_error(*this, error_context("evaluate_basis"),
        "Each coordinate tensor must be defined and 1D");
    }

    if (u.device() != options.device() || u.dtype() != options.dtype()) {
      throw utils::runtime_error(*this, error_context("evaluate_basis"),
        "All coordinate tensors must have the same dtype and device");
    }
    const auto& b = basis_[d];

    if (u.numel() > 0) {
      const auto& knotvector = b.get_knotvector();
      const int64_t p = b.get_degree();
      const double umin = knotvector[p].item<double>();
      const double umax = knotvector[knotvector.size(0) - p - 1].item<double>();
      const double umin_input = u.min().item<double>();
      const double umax_input = u.max().item<double>();

      if (umin_input < umin || umax_input > umax) {
        throw utils::runtime_error(*this, error_context("evaluate_basis"),
          "Coordinates are outside the valid parametric domain");
      }
    }

    auto spans = b.find_spans(u);
    basis_per_dim.push_back(b.evaluate(u, spans, 0));

    interleaved_shape.push_back(u.size(0));
    interleaved_shape.push_back(b.get_degree() + 1);
  }

  auto result = torch::ones(interleaved_shape, options);

  for (int64_t d = 0; d < ndim; ++d) {
    c10::SmallVector<int64_t, 6> shape(2 * ndim, 1);
    shape[2 * d] = local_coords[d].size(0);
    shape[2 * d + 1] = basis_[d].get_degree() + 1;

    result = result * basis_per_dim[d].reshape(shape);
  }

  if (is_rational_) {
    auto local_weights = get_weights().to(options);

    for (int64_t d = 0; d < ndim; ++d) {
      const auto& b = basis_[d];
      const auto& u = local_coords[d];
      const int64_t p = b.get_degree();
      const int64_t target_dim = 2 * d;

      auto spans = b.find_spans(u);
      auto j_range = torch::arange(p + 1, spans.options()).unsqueeze(0);
      auto idx = spans.unsqueeze(1) - p + j_range;

      local_weights = torch::movedim(local_weights, target_dim, 0);

      auto shape = local_weights.sizes().vec();
      shape[0] = u.size(0);
      shape.insert(shape.begin() + 1, p + 1);

      local_weights = torch::embedding(
        local_weights.reshape({local_weights.size(0), -1}), idx)
                        .reshape(shape);

      local_weights = torch::movedim(local_weights, std::vector<int64_t> {0, 1},
        std::vector<int64_t> {2 * d, 2 * d + 1});
    }

    auto numerator = result * local_weights;

    c10::SmallVector<int64_t, 3> local_axes;
    local_axes.reserve(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
      local_axes.push_back(2 * d + 1);
    }

    result = numerator / numerator.sum(local_axes, true);
  }

  std::vector<int64_t> permute_order;
  permute_order.reserve(2 * ndim);
  for (int64_t d = 0; d < ndim; ++d) {
    permute_order.push_back(2 * d);
  }
  for (int64_t d = 0; d < ndim; ++d) {
    permute_order.push_back(2 * d + 1);
  }

  return result.permute(permute_order);
}

void Patch::to_(const torch::TensorOptions& options)
{
  // Run base class first
  this->mesh::MeshBlock<Patch>::to_(options);

  // Create views for the knot vector
  int64_t idx = 0;
  for (auto& b : basis_) {
    // Length of the knot vector
    int64_t kv_size = b.get_knotvector().size(0);

    // Create a new B-Spline basis with a view of the knot vector
    b = BSplineBasis(
      meshdata_.narrow(0, idx, kv_size), b.get_degree(), true, b.get_label());
    idx += kv_size;
  }

  // Deal with the control points
  ctrlptsw_ = meshdata_.narrow(0, idx, ctrlptsw_.numel());
}

Patch::Ptr Patch::to(const torch::TensorOptions& options) const
{
  // Create a copy of this object
  auto copy = *this;
  copy.to_(options);
  return std::make_shared<Patch>(copy);
}

Patch Patch::knot_insert(
  const c10::SmallVector<torch::Tensor, 3>& new_knots, const int64_t& reps)
{
  // Validate current knots in basis
  is_finalized_or_error("knot_insertion");

  // Validate new knots
  if (new_knots.size() != get_ndim()) {
    throw utils::runtime_error(*this, error_context("knot_insertion"),
      "The length of `new_knots` must equal the patch dimension");
  }

  if (reps <= 0) {
    throw utils::runtime_error(*this, error_context("knot_insertion"),
      "Amount of times knots are inserted must be greater than 0");
  }

  // Index local variables
  const auto& options = new_knots[0].options();
  const int64_t ndim = get_ndim();
  torch::Tensor Pw = ctrlptsw_;

  for (const auto& u : new_knots) {
    // Check these are each 1-D
    if (u.ndimension() != 1 || !u.defined()) {
      throw utils::runtime_error(*this, error_context("knot_insertion"),
        "Each axis of points must be 1-dimensional and defined");
    } else if (u.device() != options.device() || u.dtype() != options.dtype()) {
      throw utils::runtime_error(*this, error_context("knot_insertion"),
        "All tensors given in `local_coords` must data type and on the same "
        "device");
    }
  }

  // Start working state from current patch
  Basis working_basis = basis_;

  // Insert knots 'reps' number of times
  for (int64_t rep = 0; rep < reps; ++rep) {
    Basis new_basis; // reset every rep

    // Insert knots in each parametric dimension
    for (int64_t p = 0; p < ndim; ++p) {
      const torch::Tensor& X_raw = new_knots[p];
      torch::Tensor Ubar;

      // Verify X is not empty
      if (X_raw.numel() != 0) {
        auto& B = working_basis[p]; // use updated basis, not basis_
        const auto [X, _] = torch::sort(X_raw);

        // Initialize a and b from Algorithim 5.5 from NURBS book
        int64_t a = B.find_spans(X[0]).item<int64_t>();
        int64_t b = B.find_spans(X[-1]).item<int64_t>();
        const torch::Tensor& U = B.get_knotvector();

        // Verify not on CUDA
        if (X.device().is_cuda()) {
          throw utils::runtime_error(
            *this, error_context("knot_insertion"), "CUDA not implemented yet");
        } else {
          auto [Pw_p, Ubar_p] =
            patch::launch_knot_insert_cpu(Pw, U, X, B.get_degree(), a, b, p);
          Pw = Pw_p;
          Ubar = Ubar_p;
        }
      } else {
        Ubar = working_basis[p].get_knotvector();
      }

      new_basis.push_back(BSplineBasis(Ubar, working_basis[p].get_degree()));
    }

    working_basis = new_basis; // carry forward for next rep
  }

  // Finalize result
  auto result = Patch(Pw, working_basis, is_rational_, label_.to_string());
  result.finalize();
  return result;
}

// In place version of knot insertion
void Patch::knot_insert_(
  const c10::SmallVector<torch::Tensor, 3>& new_knots, const int64_t& reps)
{
  // Replace current patch with patch produced by knot_insert
  *this = knot_insert(new_knots, reps);
}

std::vector<Patch::Ptr> Patch::split_elements() const
{
  is_finalized_or_error("split_elements");

  const auto& source = get_fixed_source();
  if (source.has_value() && source->projected_isotropic_source.has_value()) {
    throw utils::runtime_error(*this, error_context("split_elements"),
      "A projected fixed source is tied to this patch's basis and cannot be "
      "carried over to its elements");
  }

  const int64_t ndim = get_ndim();
  const auto options = ctrlptsw_.options();

  // Raise every interior knot to multiplicity p + 1
  c10::SmallVector<torch::Tensor, 3> new_knots;
  std::vector<int64_t> num_elements;
  std::vector<int64_t> degrees;
  bool needs_insert = false;
  for (int64_t d = 0; d < ndim; d++) {
    const auto& b = basis_[d];
    const int64_t p = b.get_degree();
    auto [unique, mult] = b.get_unique_knots_and_multiplicity();
    auto interior = unique.slice(0, 1, -1).to(options);
    auto reps = (p + 1 - mult.slice(0, 1, -1)).clamp_min(0).to(torch::kLong);
    new_knots.push_back(
      torch::repeat_interleave(interior, reps.to(interior.device())));
    needs_insert |= new_knots.back().numel() > 0;
    num_elements.push_back(unique.size(0) - 1);
    degrees.push_back(p);
  }
  Patch refined = *this;
  if (needs_insert) {
    refined = refined.knot_insert(new_knots, 1);
  }
  const auto& Pw = refined.get_ctrlptsw();

  // One Bernstein basis per dimension
  Basis element_basis;
  for (int64_t d = 0; d < ndim; d++) {
    const int64_t p = degrees[d];
    auto knots = torch::cat(
      {torch::zeros({p + 1}, options), torch::ones({p + 1}, options)});
    element_basis.push_back(BSplineBasis(knots, p));
  }

  int64_t total = 1;
  for (auto e : num_elements)
    total *= e;
  std::vector<Ptr> elements;
  elements.reserve(total);
  for (int64_t flat = 0; flat < total; flat++) {
    torch::Tensor block = Pw;
    int64_t rem = flat;
    for (int64_t d = ndim - 1; d >= 0; d--) {
      const int64_t e = rem % num_elements[d];
      rem /= num_elements[d];
      block = block.narrow(d, e * (degrees[d] + 1), degrees[d] + 1);
    }
    auto element =
      Patch::create(block.contiguous().clone(), element_basis, is_rational_);
    element->set_fill_id(get_fill_id());
    if (source.has_value()) {
      element->set_source(*source);
    }
    element->finalize();
    elements.push_back(std::move(element));
  }
  return elements;
}

// =================================================================
// Public Getters / Setters
const torch::Tensor& Patch::get_ctrlptsw() const
{
  if (!ctrlptsw_.defined()) {
    utils::runtime_error(
      *this, error_context("get_ctrlpts"), "No control points were defined");
  }

  return ctrlptsw_;
}

torch::Tensor Patch::get_ctrlpts() const
{
  if (!ctrlptsw_.defined()) {
    utils::runtime_error(
      *this, error_context("get_ctrlpts"), "No control points were defined");
  }

  if (!is_rational_) {
    return ctrlptsw_;
  }

  // Remove weights from control points
  auto ctrlptsw = ctrlptsw_.slice(-1, 0, -1);
  auto weights = ctrlptsw_.slice(-1, -1);
  return ctrlptsw / weights;
}

torch::Tensor Patch::get_weights() const
{
  if (!ctrlptsw_.defined()) {
    utils::runtime_error(*this, error_context("get_ctrlpts"),
      "No control points or weights were defined");
  }

  // Return ones if B-Spline
  if (!is_rational_) {
    auto cshape = ctrlptsw_.sizes().vec();
    cshape.pop_back();
    return torch::ones(cshape, ctrlptsw_.options());
  }

  return ctrlptsw_.slice(-1, -1).squeeze(-1);
}

int64_t Patch::get_order(size_t dim) const
{
  if (basis_.empty()) {
    throw utils::runtime_error(
      *this, error_context("get_order"), "No basis was defined");
  } else if (dim >= basis_.size()) {
    throw utils::runtime_error(*this, error_context("get_order"),
      "`dim` is outside the basis vector of length " +
        std::to_string(basis_.size()));
  }

  return basis_[dim].get_order();
}

std::vector<int64_t> Patch::get_orders() const
{
  if (basis_.empty()) {
    throw utils::runtime_error(
      *this, error_context("get_orders"), "No basis was defined");
  }

  std::vector<int64_t> orders;
  orders.reserve(basis_.size());

  for (const auto& b : basis_) {
    orders.push_back(b.get_order());
  }

  return orders;
}

int64_t Patch::get_degree(size_t dim) const
{
  if (basis_.empty()) {
    throw utils::runtime_error(
      *this, error_context("get_degree"), "No basis was defined");
  } else if (dim >= basis_.size()) {
    throw utils::runtime_error(*this, error_context("get_order"),
      "`dim` is outside the basis vector of length " +
        std::to_string(basis_.size()));
  }

  return basis_[dim].get_degree();
}

std::vector<int64_t> Patch::get_degrees() const
{
  if (basis_.empty()) {
    throw utils::runtime_error(
      *this, error_context("get_degrees"), "No basis was defined");
  }

  std::vector<int64_t> degrees;
  degrees.reserve(basis_.size());

  for (const auto& b : basis_) {
    degrees.push_back(b.get_degree());
  }

  return degrees;
}

int64_t Patch::get_ctrlpts_size(size_t dim) const
{
  if (!ctrlptsw_.defined()) {
    throw utils::runtime_error(*this, error_context("get_ctrlpts_size"),
      "No control points were defined");
  } else if (dim >= ctrlptsw_.ndimension() - 1) {
    throw utils::runtime_error(*this, error_context("get_ctrlpts_size"),
      "`dim` is outside the spline dimensionality of " +
        std::to_string(ctrlptsw_.ndimension() - 1));
  }

  return ctrlptsw_.size(dim);
}

std::vector<int64_t> Patch::get_ctrlpts_sizes() const
{
  if (!ctrlptsw_.defined()) {
    throw utils::runtime_error(*this, error_context("get_ctrlpts_sizes"),
      "No control points were defined");
  }

  auto cshape = ctrlptsw_.sizes().vec();
  if (is_rational_) {
    cshape[cshape.size() - 1] -= 1;
  }
  return cshape;
}

torch::Tensor Patch::get_bbox_impl(double epsilon) const
{
  // Strong convex hull property
  const auto& ctrlpts = get_ctrlpts();

  // Find the minimum point and maximum bounding the whole patch
  // epsilon is a little buffer
  auto flat_view = ctrlpts.reshape({-1, ctrlpts.size(-1)});
  auto min_point = std::get<0>(flat_view.min(0)) - epsilon;
  auto max_point = std::get<0>(flat_view.max(0)) + epsilon;

  // Return 2xP tensor where P is the number of physical dimensions
  return torch::stack({min_point, max_point}, 0);
}

Patch Patch::get_boundary_impl(size_t dim, bool is_upper)
{
  is_finalized_or_error("get_boundary");

  // Check the dim is valid
  if (dim >= ctrlptsw_.ndimension() - 1) {
    throw utils::runtime_error(*this, error_context("get_boundary"),
      "`dim` is outside the spline dimensionality of " +
        std::to_string(ctrlptsw_.ndimension() - 1));
  }

  // Create a new patch
  Patch new_patch = *this;

  // Control points for the boundary (this will produce a non-contiguous array
  // so we will need to factor that into any methods)
  new_patch.ctrlptsw_ =
    ctrlptsw_.select(dim, is_upper ? (ctrlptsw_.size(dim) - 1) : 0);

  // Remove the basis
  new_patch.basis_.erase(new_patch.basis_.begin() + dim);

  // We label this as a clone
  new_patch.label_ = label_.clone();

  return new_patch;
}

int64_t Patch::get_numel_impl(size_t dim) const
{
  // Check the dim is valid
  if (dim >= basis_.size()) {
    throw utils::runtime_error(*this, error_context("get_numel"),
      "`dim` is outside the spline dimensionality of " +
        std::to_string(basis_.size()));
  }

  return basis_[dim].get_unique_knots().size(0) - 1;
}

void Patch::set_ctrlptsw(const torch::Tensor& ctrlptsw)
{
  is_not_finalized_or_error("set_ctrlptsw");
  ctrlptsw_ = ctrlptsw.clone();
  is_rational_ = true;
}

void Patch::set_basis(const Basis& basis)
{
  is_not_finalized_or_error("set_basis");
  basis_ = basis;
}

void Patch::set_ctrlpts(torch::Tensor ctrlpts)
{
  is_not_finalized_or_error("set_ctrlpts");

  if (!ctrlptsw_.defined() || !is_rational_) {
    ctrlptsw_ = ctrlpts.clone();

  } else {
    auto cshape = ctrlptsw_.sizes().vec();
    cshape[cshape.size() - 1] -= 1;

    if (cshape != ctrlpts.sizes().vec()) {
      // Control point shape is different, reset to B-Spline
      ctrlptsw_ = ctrlpts.clone();
      is_rational_ = false;

    } else {
      // Weight and set the control points for the NURBS
      ctrlptsw_.slice(-1, 0, -1).copy_(
        ctrlpts.to(ctrlptsw_.options()) * ctrlptsw_.slice(-1, -1));
    }
  }

  // Ensure the array is contiguous
  ctrlptsw_ = ctrlptsw_.contiguous();
}

int64_t Patch::get_num_dofs_impl() const
{
  return is_finalized_ ? (is_rational_ ? ctrlptsw_.slice(-1, 0, -1).numel()
                                       : ctrlptsw_.numel())
                       : 0;
}

void Patch::set_weights(torch::Tensor weights)
{
  is_not_finalized_or_error("set_weights");

  // Check that the control points have been defined first
  if (!ctrlptsw_.defined()) {
    throw utils::runtime_error(*this, error_context("set_weights"),
      "Set the control points before setting the weights");
  }

  // Add dimension for broadcasting
  if (weights.ndimension() == ctrlptsw_.ndimension() - 1) {
    weights = weights.unsqueeze(-1);
  }

  // Check the shape of the weights tensor
  auto cshape = ctrlptsw_.sizes().vec();
  cshape.pop_back();
  auto wshape = weights.sizes().vec();
  wshape.pop_back();
  if (cshape != wshape || weights.size(weights.ndimension() - 1) != 1) {
    throw utils::runtime_error(*this, error_context("set_weights"),
      "`weights.shape` must match `get_ctrlpts().shape[:-1]` or "
      "`(*get_ctrlpts().shape[:-1], 1)`");
  }

  // Make sure device and data type is consistent
  weights = weights.to(ctrlptsw_.options());

  if (!is_rational_) {
    // Convert to NURBS
    ctrlptsw_ = torch::cat({ctrlptsw_ * weights, weights}, -1);
    is_rational_ = true;

  } else {
    // Update the weights based on the old ones
    auto w_old = ctrlptsw_.slice(-1, -1);
    ctrlptsw_.slice(-1, 0, -1).mul_(weights / w_old);
    w_old.copy_(weights);
  }

  // Ensure the array is contiguous
  ctrlptsw_ = ctrlptsw_.contiguous();
}

std::ostream& operator<<(std::ostream& os, const Patch& p)
{
  os << "Patch(\n  ctrlpts_size=";

  if (p.get_ctrlptsw().defined()) {
    std::string sep = "(";
    for (const auto& n : p.get_ctrlpts_sizes()) {
      os << sep << n;
      sep = ", ";
    }

    auto boolean = [](const bool& b) { return b ? "True" : "False"; };
    os << "),\n  is_rational=" << boolean(p.is_rational())
       << ",\n  is_finalized=" << boolean(p.is_finalized())
       << ",\n  device=" << p.get_ctrlptsw().device()
       << ",\n  dtype=" << p.get_ctrlptsw().scalar_type()
       << ",\n  label=" << p.get_label() << ",\n  basis=";

  } else {
    os << "None,\n  is_rational=False,\n  is_valid=False,\n  device=None,\n "
          "dtype=None,\n  basis=";
  }

  if (!p.get_basis().empty()) {
    os << "[\n";

    for (const auto& b : p.get_basis()) {
      std::stringstream ss;
      ss << b;
      os << utils::indent_message(ss.str(), 4) << ",\n";
    }

    os << "  ]\n)";
  } else {
    os << "None)";
  }

  return os;
}

} // namespace ttnte::cad
