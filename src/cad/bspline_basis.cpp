#include "ttnte/cad/bspline_basis.hpp"
#include "ttnte/cad/bspline_kernels.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::cad {

// =================================================================
// Public constructors
BSplineBasis::BSplineBasis(const torch::Tensor& knotvector, int64_t degree,
  std::optional<std::string> label)
  : Basis<BSplineBasis>(label), knotvector_(knotvector.clone()), degree_(degree)
{
  normalize_knotvector();
}
BSplineBasis::BSplineBasis(
  const torch::Tensor& knotvector, int64_t degree, const Label& label)
  : Basis<BSplineBasis>(label), knotvector_(knotvector.clone()), degree_(degree)
{
  normalize_knotvector();
}

BSplineBasis::BSplineBasis(const torch::Tensor& knotvector, int64_t degree,
  bool is_finalized, const Label& label)
  : Basis<BSplineBasis>(label), degree_(degree)
{
  if (is_finalized) {
    knotvector_ = knotvector;
    is_finalized_ = true;
  } else {
    knotvector_ = knotvector.clone();
    normalize_knotvector();
    is_finalized = false;
  }
}

// =================================================================
// Public methods
void BSplineBasis::finalize(const torch::Tensor& knotvector_view)
{
  if (is_finalized_) {
    throw utils::runtime_error(*this, error_context("finalize"),
      "This basis has already been finalized");
  }

  // Copy view
  knotvector_ = knotvector_view;

  if (knotvector_.ndimension() != 1) {
    throw utils::runtime_error(*this, error_context("is_valid"),
      "The knot vector must be 1-dimensional");
  }

  if (degree_ < 1) {
    throw utils::runtime_error(*this, error_context("is_valid"),
      "The degree must be an integer greater than or equal to 1");
  }

  // Check if the given array is ascending
  auto left = knotvector_.slice(0, 0, -1);
  auto right = knotvector_.slice(0, 1, knotvector_.size(0));

  if (!torch::all(left <= right).item<bool>()) {
    throw utils::runtime_error(*this, error_context("is_valid"),
      "Knots in the knot vector must be in ascending order");
  }

  is_finalized_ = true;
}

void BSplineBasis::normalize_knotvector()
{
  knotvector_ =
    (knotvector_ - knotvector_[0]) / knotvector_[knotvector_.size(0) - 1];
}

BSplineBasis& BSplineBasis::to_(
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  auto options = knotvector_.options();

  if (device.has_value()) {
    options = options.device(device.value());
  }
  if (dtype.has_value()) {
    options = options.dtype(dtype);
  }

  knotvector_.to(options);

  return *this;
}

BSplineBasis BSplineBasis::to(std::optional<torch::Device> device,
  std::optional<torch::ScalarType> dtype) const
{
  auto new_patch = *this;
  return std::move(new_patch.to(device, dtype));
}

BSplineBasis BSplineBasis::clone() const
{
  return BSplineBasis(knotvector_, degree_, std::move(label_.clone()));
}

torch::Tensor BSplineBasis::find_spans(const torch::Tensor& u) const
{
  // Find start of span threshold index in knotvector
  auto span_idxs =
    torch::searchsorted(knotvector_.to(u.options()), u, false, true) - 1;
  return torch::clamp(span_idxs, degree_, get_size() - 1);
}

torch::Tensor BSplineBasis::evaluate(
  const torch::Tensor& u, const int64_t& derivative_order) const
{
  return evaluate(u, find_spans(u), derivative_order);
}

torch::Tensor BSplineBasis::evaluate(const torch::Tensor& u,
  const torch::Tensor& spans, const int64_t& derivative_order) const
{
  auto options = u.options();

  if (u.device().is_cuda()) {
    throw utils::runtime_error(
      *this, error_context("evaluate"), "CUDA not implemented yet");
  } else {
    return bspline::launch_evaluate_cpu(u, spans.to(u.device()),
      knotvector_.to(u.options()), degree_, derivative_order);
  }
}

torch::Tensor BSplineBasis::evaluate_all(
  const torch::Tensor& u, const int64_t& derivative_order) const
{
  return evaluate_all(u, find_spans(u), derivative_order);
}

torch::Tensor BSplineBasis::evaluate_all(const torch::Tensor& u,
  const torch::Tensor& spans, const int64_t& derivative_order) const
{
  // Evaluate non-zero basis functions
  auto nonzero = evaluate(u, spans, derivative_order);

  // Create column indices for the scatter operation: shape [Q, p+1]
  // For each point, the active columns are: span - p + k (where k is 0 to p)
  torch::Tensor k_idx = torch::arange(0, degree_ + 1, spans.options())
                          .unsqueeze(0); // (1, degree_ + 1)
  torch::Tensor col_indices =
    spans.unsqueeze(1) - degree_ + k_idx; // (u.size(0), degree_ + 1)

  // Scatter the local non-zero values into the dense matrix
  if (derivative_order == 0) {
    // No derivatives
    auto result = torch::zeros({u.size(0), get_size()}, nonzero.options());

    result.scatter_(1, col_indices, nonzero);
    return result;

  } else {
    // Handle derivatives
    auto result = torch::zeros(
      {u.size(0), std::min(derivative_order, degree_) + 1, get_size()},
      nonzero.options());

    for (int64_t k = 0; k <= std::min(derivative_order, degree_); ++k) {
      // Scatter into the corresponding 2D slice of the result
      result.select(1, k).scatter_(1, col_indices, nonzero.select(1, k));
    }

    return result;
  }
}

void BSplineBasis::pack(std::vector<int64_t>& meta_buffer,
  std::vector<torch::Tensor>& payload_buffer) const
{
  // Fill the meta data buffer first
  meta_buffer.push_back(label_.to_int());     // BSplineBasis label
  meta_buffer.push_back(knotvector_.size(0)); // Size of the knot vector
  meta_buffer.push_back(degree_);             // Polynomial degree

  // Add knot vector to the payload
  payload_buffer.push_back(knotvector_.contiguous());
}

std::ostream& operator<<(std::ostream& os, const BSplineBasis& p)
{
  os << "BSplineBasis(\n  knotvector=[";
  const auto& knotvector = p.get_knotvector();

  // Print knot vector
  AT_DISPATCH_FLOATING_TYPES(
    knotvector.scalar_type(), "bspline_basis_repr", [&] {
      std::string sep = "";
      for (size_t i = 0; i < knotvector.size(0); i++) {
        os << sep << knotvector[i].item<scalar_t>();
        sep = ", ";
      }
    });

  os << "],\n  degree=" << p.get_degree() << ",\n  device=" << p.get_device()
     << ",\n  dtype=" << p.get_dtype() << ",\n  label=" << p.get_label()
     << "\n)";
  return os;
}

} // namespace ttnte::cad
