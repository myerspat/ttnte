#include "ttnte/cad/interface_quadrature.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ttnte::cad {

namespace {

/// Breakpoints of the two faces closer than this (relative to [0, 1]) are
/// one: the mesh matches geometry at the same tolerance, and a sliver span
/// would only cost points.
constexpr double BREAKPOINT_TOL = 1e-8;

/// @return The unique knots of a basis scaled to [0, 1] (reversed if
/// `flip`), and its parametric range.
std::vector<double> normalized_knots(
  const BSplineBasis& basis, bool flip, double& lo, double& hi)
{
  auto knots = basis.get_unique_knots().to(torch::kFloat64).cpu().contiguous();
  const double* k = knots.data_ptr<double>();
  const int64_t n = knots.numel();
  lo = k[0];
  hi = k[n - 1];
  if (n < 2 || !(hi > lo))
    throw utils::runtime_error("ttnte::cad::interface_quadrature",
      "A face basis has an empty parametric range");
  std::vector<double> out(n);
  for (int64_t i = 0; i < n; i++) {
    const double s = (k[i] - lo) / (hi - lo);
    out[i] = flip ? 1.0 - s : s;
  }
  std::sort(out.begin(), out.end());
  return out;
}

} // namespace

InterfaceQuadrature interface_quadrature(const std::vector<BSplineBasis>& first,
  const std::vector<BSplineBasis>& second,
  const std::vector<int64_t>& second_dim, const std::vector<bool>& flip)
{
  const std::string ctx = "ttnte::cad::interface_quadrature";
  const int64_t K = static_cast<int64_t>(first.size());
  if (static_cast<int64_t>(second.size()) != K ||
      static_cast<int64_t>(second_dim.size()) != K ||
      static_cast<int64_t>(flip.size()) != K || K < 1) {
    throw utils::runtime_error(
      ctx, "Both faces need the same number (>= 1) of tangential dimensions");
  }
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);

  InterfaceQuadrature out;
  out.second_dim = second_dim;
  out.flip = flip;
  for (int64_t k = 0; k < K; k++) {
    const auto& ba = first[k];
    const auto& bb = second.at(second_dim[k]);
    double lo_a, hi_a, lo_b, hi_b;
    auto ka = normalized_knots(ba, false, lo_a, hi_a);
    auto kb = normalized_knots(bb, flip[k], lo_b, hi_b);

    // Union of the breakpoints
    std::vector<double> breaks(ka);
    breaks.insert(breaks.end(), kb.begin(), kb.end());
    std::sort(breaks.begin(), breaks.end());
    std::vector<double> unique;
    for (double x : breaks)
      if (unique.empty() || x - unique.back() > BREAKPOINT_TOL)
        unique.push_back(x);
    unique.front() = 0.0;
    unique.back() = 1.0;

    // max(p) + 1 Gauss-Legendre points per sub-span, mapped as the face
    // quadratures of the sweep and TT assemblers map theirs
    const int64_t p = std::max(ba.get_degree(), bb.get_degree());
    auto [s, ws] = math::QuadratureSet1D::gauss_legendre(p + 1)->map_to_spans(
      torch::tensor(unique, opts));

    out.points_first.push_back(lo_a + (hi_a - lo_a) * s);
    out.weights_first.push_back((hi_a - lo_a) * ws);
    out.points_second.push_back(lo_b + (hi_b - lo_b) * (flip[k] ? 1.0 - s : s));
  }

  // Tensor grid over k (C order)
  auto grid = [&](const std::vector<torch::Tensor>& per_k) {
    auto mesh = torch::meshgrid(per_k, "ij");
    std::vector<torch::Tensor> cols;
    for (auto& m : mesh)
      cols.push_back(m.flatten());
    return torch::stack(cols, 1);
  };
  out.u_first = grid(out.points_first).contiguous();
  out.weights = grid(out.weights_first).prod(1).contiguous();
  auto ub = grid(out.points_second);

  // Columns of the second face in its own tangential order
  std::vector<torch::Tensor> cols(K);
  for (int64_t k = 0; k < K; k++)
    cols.at(second_dim[k]) = ub.select(1, k);
  out.u_second = torch::stack(cols, 1).contiguous();
  return out;
}

InterfaceQuadrature interface_quadrature(const Patch& first, int64_t dim_first,
  const Patch& second, int64_t dim_second, const mesh::BoundaryMapping& mapping)
{
  const auto& bases_a = first.get_basis();
  const auto& bases_b = second.get_basis();
  const int64_t ndim = static_cast<int64_t>(bases_a.size());
  if (static_cast<int64_t>(bases_b.size()) != ndim || ndim < 2 ||
      static_cast<int64_t>(mapping.perm.size()) != ndim ||
      static_cast<int64_t>(mapping.flip.size()) != ndim - 1) {
    throw utils::runtime_error("ttnte::cad::interface_quadrature",
      "The patches and the mapping must share the dimension (>= 2)");
  }
  std::vector<BSplineBasis> ta, tb;
  for (int64_t d = 0; d < ndim; d++) {
    if (d != dim_first)
      ta.push_back(bases_a[d]);
    if (d != dim_second)
      tb.push_back(bases_b[d]);
  }
  auto [second_dim, flip] =
    tangential_mapping(ndim, dim_first, dim_second, mapping);
  return interface_quadrature(ta, tb, second_dim, flip);
}

void check_shared_parametrization(const Patch& face_first,
  const Patch& face_second, const InterfaceQuadrature& iq, double tol)
{
  const auto xa = face_first.evaluate(iq.u_first.to(face_first.get_dtype()));
  const auto xb =
    face_second.evaluate(iq.u_second.to(face_second.get_dtype())).to(xa);
  const double scale = 1.0 + xa.abs().max().item<double>();
  if ((xa - xb).abs().max().item<double>() > tol * scale) {
    throw utils::runtime_error("ttnte::cad::check_shared_parametrization",
      "The faces of a non-conforming interface do not share their "
      "parametrization (refine both from one patch by knot insertion and "
      "degree elevation)");
  }
}

std::pair<std::vector<int64_t>, std::vector<bool>> tangential_mapping(
  int64_t ndim, int64_t dim_first, int64_t dim_second,
  const mesh::BoundaryMapping& mapping)
{
  // The second patch's dimension along each tangential dimension of the
  // first, renumbered among the second face's tangential dimensions
  std::vector<int64_t> second_dim;
  std::vector<bool> flip;
  int64_t k = 0;
  for (int64_t d = 0; d < ndim; d++) {
    if (d == dim_first)
      continue;
    const int64_t db = mapping.perm[d];
    second_dim.push_back(db - (db > dim_second ? 1 : 0));
    flip.push_back(mapping.flip[k++]);
  }
  return {second_dim, flip};
}

} // namespace ttnte::cad
