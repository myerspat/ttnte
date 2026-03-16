#include "ttnte/cad/patch.hpp"
#include "ttnte/cad/bspline_basis.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/io_formatting.hpp"

namespace ttnte::cad {

// =================================================================
// Public constructors
Patch::Patch(std::optional<std::string> label)
  : label_(
      label.has_value() ? Label::from_string(*label) : Label::create_internal())
{
  invalidate();
}

Patch::Patch(const torch::Tensor& ctrlpts, const Basis& basis, bool is_rational,
  std::optional<std::string> label)
  : is_rational_(is_rational),
    label_(
      label.has_value() ? Label::from_string(*label) : Label::create_internal())
{
  set_basis(basis, true);
  set_ctrlptsw(ctrlpts, true);

  // Check this is a valid patch
  validate();
}

Patch::Patch(const torch::Tensor& ctrlpts, const torch::Tensor& weights,
  const Basis& basis, std::optional<std::string> label)
  : label_(
      label.has_value() ? Label::from_string(*label) : Label::create_internal())
{
  set_basis(basis, true);
  set_ctrlpts(ctrlpts, true);
  set_weights(weights, true);

  // Check this is a valid patch
  validate();
}

// =================================================================
// Public methods
void Patch::validate()
{
  if (!is_valid_) {
    if (basis_.empty() || !ctrlptsw_.defined()) {
      throw utils::runtime_error(*this, error_context("validate"),
        "Both the basis and control points must be set before validation");
    }

    // Check the dimensionality
    if (basis_.size() != ctrlptsw_.ndimension() - 1) {
      std::stringstream ss;
      ss << basis_.size() << "-dimensional basis expects a "
         << basis_.size() + 1 << "-dimensional control point tensor\n"
         << "A " << ctrlptsw_.ndimension()
         << "-dimensional control point tensor was given";

      throw utils::runtime_error(*this, error_context("validate"), ss.str());
    }

    // Check the control point tensor shape matches the basis
    for (size_t i = 0; i < basis_.size(); i++) {
      if (basis_[i].get_size() != ctrlptsw_.size(i)) {
        throw utils::runtime_error(*this, error_context("validate"),
          "There must be an equal number of basis functions and control points"
          "along each dimension");
      }
    }

    // Check device and data type
    auto device = ctrlptsw_.device();
    auto dtype = ctrlptsw_.dtype();
    for (const auto& b : basis_) {
      if (b.get_device() != device || b.get_dtype() != dtype) {
        throw utils::runtime_error(*this, error_context("validate"),
          "The basis and control points should be on the same device with the "
          "same data type");
      }
    }

    is_valid_ = true;
  }
}

Patch Patch::clone() const
{
  c10::SmallVector<BSplineBasis> basis;
  basis.reserve(basis_.size());

  for (const auto& b : basis_) {
    basis.push_back(b.clone());
  }

  return Patch(ctrlptsw_.clone(), std::move(basis), is_rational_, is_valid_,
    label_.clone());
}

torch::Tensor Patch::evaluate(const torch::Tensor& local_coords)
{
  // Check the patch is valid
  validate();

  // Check input
  if (local_coords.ndimension() != 2) {
    throw utils::runtime_error(
      *this, error_context("evaluate"), "`local_coords` must be 2-dimensional");
  } else if (local_coords.size(1) != basis_.size()) {
    throw utils::runtime_error(*this, error_context("evaluate"),
      "The second dimension of `local_coords` should be size " +
        std::to_string(basis_.size()));
  }

  // Apply one basis at a time
  int64_t n = local_coords.size(0);
  int64_t phys_d = ctrlptsw_.size(-1);
  auto phys_coords = ctrlptsw_.to(local_coords.options());

  for (size_t d = 0; d < basis_.size(); d++) {
    const auto& b = basis_[d];
    const auto& p = b.get_degree();
    const auto& u = local_coords.select(1, d);

    // Find spans for each coordinate
    auto spans = b.find_spans(u);

    // Gather indices: spans - p + j
    auto j_range = torch::arange(p + 1, spans.options()).unsqueeze(0);
    auto idx = spans.unsqueeze(1) - p + j_range;

    if (d == 0) {
      // Ending shape: [N_1, ..., N_D, phys_dims] -> [n, N_2, ..., N_D,
      // phys_dims]
      auto shape = phys_coords.sizes().vec();
      shape[0] = -1;

      // Flatten the rest of the dimensions: [b.get_size(), -1]
      auto phys_coords_flat = phys_coords.view({phys_coords.size(0), -1});

      // Multiply by basis functions and sum across the p + 1 dimension after
      // embedding to create [n, p + 1, -1]
      // [n, 1, p + 1] @ [n, p + 1, m] -> [n, 1, m]
      phys_coords = torch::matmul(b.evaluate(u, spans).unsqueeze(1),
        torch::embedding(phys_coords_flat, idx));

      // Reshape to inject our batch dimension: [N_points, N_2, ..., phys_dims]
      phys_coords = phys_coords.view(shape);

    } else {
      // Ending shape: [n, N_d, ..., N_D, phys_dims]
      // -> [n, N_(d + 1), ..., N_D, phys_dims]
      auto shape = phys_coords.sizes().vec();
      shape.erase(shape.begin() + 1);

      // Note n_b = b.get_size()
      int64_t n_b = phys_coords.size(1);

      // Reshape to expose the target dimension
      auto phys_coords_flat = phys_coords.view({n, n_b, -1});

      // Multiply by basis functions and sum
      // [n, 1, p + 1] @ [n, p + 1, m] -> [n, 1, m]
      phys_coords = torch::matmul(b.evaluate(u, spans).unsqueeze(1),
        torch::gather(phys_coords_flat, /*dim=*/1,
          idx.unsqueeze(2).expand({n, p + 1, phys_coords_flat.size(-1)})));

      // Reshape back: [N_points, N_{d+1}, ..., phys_dims]
      phys_coords = phys_coords.view(shape);
    }
  }

  return is_rational_ ? phys_coords.slice(1, 0, -1) / phys_coords.slice(1, -1)
                      : phys_coords;
}

torch::Tensor Patch::evaluate(
  const c10::SmallVector<torch::Tensor, 3>& local_coords)
{
  // Check the patch is valid
  validate();

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

// =================================================================
// Public Getters / Setters
torch::Tensor Patch::get_ctrlpts() const
{
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

torch::Device Patch::get_device()
{
  validate();
  return ctrlptsw_.device();
}

torch::ScalarType Patch::get_dtype()
{
  validate();
  return ctrlptsw_.scalar_type();
}

void Patch::set_ctrlptsw(const torch::Tensor& ctrlptsw, bool clone)
{
  ctrlptsw_ = clone ? ctrlptsw.clone().contiguous() : ctrlptsw.contiguous();
  invalidate();
}

torch::Tensor Patch::get_bbox(double epsilon) const
{
  // Strong convex hull property
  const auto& ctrlpts = get_ctrlpts();

  // Find the minimum point and maximum bounding the whole patch
  // epsilon is a little buffer
  auto flat_view = ctrlpts.view({-1, ctrlpts.size(-1)});
  auto min_point = std::get<0>(flat_view.min(0)) - epsilon;
  auto max_point = std::get<0>(flat_view.max(0)) + epsilon;

  // Return 2xP tensor where P is the number of physical dimensions
  return torch::stack({min_point, max_point}, 0);
}

Patch Patch::get_boundary(size_t dim, bool is_upper, bool clone)
{
  // Check we have a valid patch first
  validate();

  // Check the dim is valid
  if (dim >= ctrlptsw_.ndimension() - 1) {
    throw utils::runtime_error(*this, error_context("get_boundary"),
      "`dim` is outside the spline dimensionality of " +
        std::to_string(ctrlptsw_.ndimension() - 1));
  }

  // Control points for the boundary
  torch::Tensor boundary_ctrlpts =
    ctrlptsw_.select(dim, is_upper ? (ctrlptsw_.size(dim) - 1) : 0);

  // Get the correct basis
  Basis boundary_basis;
  boundary_basis.reserve(basis_.size() - 1);

  for (size_t i = 0; i < basis_.size(); i++) {
    if (i == dim)
      continue;
    boundary_basis.push_back(basis_[i]);
  }

  // Clone if needed
  if (clone) {
    boundary_ctrlpts = boundary_ctrlpts.clone();

    for (auto& b : boundary_basis) {
      b = b.clone();
    }

    return Patch(boundary_ctrlpts, boundary_basis, is_rational_, is_valid_,
      label_.clone());
  }

  return Patch(
    boundary_ctrlpts, boundary_basis, is_rational_, is_valid_, label_);
}

void Patch::set_basis(const Basis& basis, bool clone)
{
  if (clone) {
    Basis new_basis;
    new_basis.reserve(basis.size());

    for (const auto& b : basis) {
      new_basis.push_back(b.clone());
    }

    basis_ = new_basis;

  } else {
    basis_ = basis;
  }

  invalidate();
}

void Patch::set_ctrlpts(torch::Tensor ctrlpts, bool clone)
{
  if (clone) {
    ctrlpts = ctrlpts.clone();
  }

  if (!ctrlptsw_.defined() || !is_rational_) {
    ctrlptsw_ = clone ? ctrlpts : ctrlpts;

  } else {
    auto cshape = ctrlptsw_.sizes().vec();
    cshape[cshape.size() - 1] -= 1;

    if (cshape != ctrlpts.sizes().vec()) {
      // Control point shape is different, reset to B-Spline
      ctrlptsw_ = ctrlpts;
      is_rational_ = false;

    } else {
      // Weight and set the control points for the NURBS
      ctrlptsw_.slice(-1, 0, -1).copy_(
        ctrlpts.to(ctrlptsw_.options()) * ctrlptsw_.slice(-1, -1));
    }
  }

  // Ensure the array is contiguous
  ctrlptsw_ = ctrlptsw_.contiguous();
  invalidate();
}

void Patch::set_weights(torch::Tensor weights, bool clone)
{
  if (clone) {
    weights = weights.clone();
  }

  // Check that the control points have been defined first
  if (!ctrlptsw_.defined()) {
    throw utils::runtime_error(*this, error_context("set_weights"),
      "Set the control points before setting the weights");
  }

  // Add dimension for broadcasting
  if (weights.ndimension() == ctrlptsw_.ndimension() - 1) {
    weights.unsqueeze_(-1);
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
    ctrlptsw_.mul_(weights);
    ctrlptsw_ = torch::cat({ctrlptsw_, weights}, -1);
    is_rational_ = true;

  } else {
    // Update the weights based on the old ones
    auto w_old = ctrlptsw_.slice(-1, -1);
    ctrlptsw_.slice(-1, 0, -1).mul_(weights / w_old);
    w_old.copy_(weights);
  }

  // Ensure the array is contiguous
  ctrlptsw_ = ctrlptsw_.contiguous();
  invalidate();
}

std::ostream& operator<<(std::ostream& os, const Patch& p)
{
  os << "Patch(\n  ctrlpts_size=";

  auto boolean = [](const bool& b) { return b ? "True" : "False"; };

  if (p.get_ctrlptsw().defined()) {
    std::string sep = "(";
    for (const auto& n : p.get_ctrlpts_sizes()) {
      os << sep << n;
      sep = ", ";
    }
    os << "),\n  is_rational=" << boolean(p.is_rational())
       << ",\n  is_valid=" << boolean(p.is_valid())
       << ",\n  device=" << p.get_ctrlptsw().device()
       << ",\n  dtype=" << p.get_ctrlptsw().scalar_type()
       << ",\n  label=" << p.get_label() << ",\n  basis=";

  } else {
    os << "None,\n  is_rational=False,\n  is_valid=False,\n  device=None,\n  "
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
