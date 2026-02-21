#include <stdexcept>

#include "ttnte/cad/basis_functions.hpp"

namespace ttnte::cad {
// Constructors
BasisFunctions::BasisFunctions() : knots_{}, degrees_{} {}

BasisFunctions::BasisFunctions(
  std::vector<torch::Tensor> knots, std::vector<int64_t> degrees)
  : knots_(std::move(knots)), degrees_(std::move(degrees))
{
  if (knots_.size() != degrees_.size()) {
    throw std::invalid_argument("BasisFunctions ctor: knots and degrees must have same size");
  }
}

// Single Dimension Evaluation
torch::Tensor BasisFunctions::basis_functions_bspline(
  int64_t param_idx, const torch::Tensor & spans, const torch::Tensor & coords)
{
  const int64_t p = degrees_[param_idx];
  const auto & knots_p = knots_[param_idx];

  const int64_t num_coords = coords.size(0);
  auto basis = torch::zeros({num_coords, p + 1}, torch::kFloat64);

  const auto knots_a = knots_p.accessor<double, 1>();
  const auto spans_a = spans.accessor<int64_t, 1>();
  const auto coords_a = coords.accessor<double, 1>();
  auto basis_a = basis.accessor<double, 2>();

  // Temporary arrays for Cox-de Boor recursion
  std::vector<double> left(p + 1);
  std::vector<double> right(p + 1);

  for (int64_t idx = 0; idx < num_coords; ++idx) {
    int64_t span = spans_a[idx];
    double u = coords_a[idx];

    // Initialize
    basis_a[idx][0] = 1.0;

    // Cox-de Boor recursion
    for (int64_t j = 1; j <= p; ++j) {
      left[j] = u - knots_a[span + 1 - j];
      right[j] = knots_a[span + j] - u;

      double saved = 0.0;
      for (int64_t r = 0; r < j; ++r) {
        double temp = basis_a[idx][r] / (right[r + 1] + left[j - r]);
        basis_a[idx][r] = saved + right[r + 1] * temp;
        saved = left[j - r] * temp;
      }
      basis_a[idx][j] = saved;
    }
  }

  return basis;
}
/*
torch::Tensor BasisFunctions::basis_functions_ders_bspline(
  int64_t param_idx, torch::Tensor spans, torch::Tensor coords, int64_t order)
{

}*/

// Batch Evaluation
torch::Tensor find_spans(int64_t param_idx, const torch::Tensor & coords) {

  if (param_idx < 0 || param_idx >= degrees_.size()) {
    throw std::out_of_range("find_spans: param_idx out of range");
  }

  const auto & knots_p = knots_[param_idx];
  const int64_t deg_p = degrees_[param_idx];
  const int64_t n = (knots_p.size() - 1) - deg_p - 1;

  int64_t = num_coords = coords.size();
  auto spans = torch::empty({num_coords}, torch::dtype(torch::kInt64));

  // Access to underlying data
  const auto U = knots_p.accessor<double, 1>();
  const auto u = coords.accessor<double, 1>();
  auto s_a = spans.accessor<int64_t, 1>();

  for (int64_t idx = 0; idx < num_coords; idx++) {
    const auto & u_idx = u[idx]; // Evaluation param
    if (u_idx >= U[n + 1]) {
      s_a[idx] = n;
      continue;
    }
    int64_t low = deg_p;
    int64_t high = n + 1;
    int64_t mid = (low + high)/2;

    while (u_idx < U[mid] || u >= U[mid + 1]) {
      if (u < U[mid]) { high = mid; }
      else { low = mid; }
      mid = (low + high)/2;
    }

    s_a[idx] = mid;
  }
}

} // namespace ttnte::cad
