#include "ttnte/cad/patch.hpp"
#include "ttnte/cad/bspline_basis.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/io_formatting.hpp"
#include <sstream>

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
    throw utils::runtime_error(*this, error_context("validate"),
      "Both the basis and control points must be set before validation");
  }

  // Check the dimensionality
  if (basis_.size() != ctrlptsw_.ndimension() - 1) {
    std::stringstream ss;
    ss << basis_.size() << "-dimensional basis expects a " << basis_.size() + 1
       << "-dimensional control point tensor\n"
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
}

torch::Tensor Patch::evaluate(const torch::Tensor& local_coords)
{
  is_finalized_or_error("evaluate");

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

// void Patch::pack(std::vector<int64_t>& meta_buffer,
//   std::vector<torch::Tensor>& payload_buffer) const
// {
//   // Fill the meta data buffer first
//   meta_buffer.push_back(label_.to_int());                 // Patch label
//   meta_buffer.push_back(static_cast<int64_t>(is_valid_)); // Is valid Boolean
//   meta_buffer.push_back(
//     static_cast<int64_t>(is_rational_)); // Is rational Boolean
//   meta_buffer.push_back(basis_.size());  // Number of parametric dimensions
//
//   // Iterate through the basis
//   for (const auto& b : basis_) {
//     b.pack(meta_buffer, payload_buffer);
//   }
//
//   // Add the control point information
//   meta_buffer.push_back(ctrlptsw_.numel()); // Number of control points
//   payload_buffer.push_back(ctrlptsw_.flatten().contiguous());
// }

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

torch::Tensor Patch::evaluate_basis(
  const c10::SmallVector<torch::Tensor, 3>& local_coords)
{
  is_finalized_or_error("evaluate_basis");

  if (local_coords.size() != get_ndim()) {
    throw utils::runtime_error(*this, error_context("evaluate_basis"),
      "The length of `local_coords` must equal the patch dimension");
  }

  const auto& options = local_coords[0].options();
  const int64_t ndim = get_ndim();

  c10::SmallVector<torch::Tensor, 3> basis_per_dim;
  basis_per_dim.reserve(ndim);

  std::vector<int64_t> interleaved_shape;
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
    std::vector<int64_t> shape(2 * ndim, 1);
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

      local_weights = torch::movedim(
        local_weights, std::vector<int64_t>{0, 1},
        std::vector<int64_t>{2 * d, 2 * d + 1});
    }

    auto numerator = result * local_weights;

    std::vector<int64_t> local_axes;
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
  auto flat_view = ctrlpts.view({-1, ctrlpts.size(-1)});
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
