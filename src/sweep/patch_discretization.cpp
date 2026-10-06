#include "ttnte/sweep/patch_discretization.hpp"
#include "ttnte/linalg/banded_lu.hpp"
#include "ttnte/math/quadrature_set.hpp"
#include "ttnte/utils/exception.hpp"
#include <numeric>

namespace {

using torch::indexing::Slice;

/// @brief Per-dimension quadrature and non-vanishing B-spline data.
struct DimData {
  /// (n) parametric quadrature measure (span width x normalized GL weight).
  torch::Tensor weights;
  /// (n, 2, p + 1) non-vanishing B-spline values and first derivatives.
  torch::Tensor basis;
  /// (n) index of the first non-vanishing basis function.
  torch::Tensor first;
  /// p + 1
  int64_t num_active = 0;
};

DimData make_dim_data(const ttnte::cad::BSplineBasis& basis)
{
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  const int64_t p = basis.get_degree();

  // Gauss-Legendre with p + 1 points per knot span, exactly as
  // DGFirstOrderTransportBackend::init_spatial_quadrature_()
  auto gl = ttnte::math::QuadratureSet1D::gauss_legendre(p + 1);
  auto x = gl->get_points().to(opts);
  auto w = gl->get_weights().to(opts);

  auto knots = basis.get_unique_knots().to(opts);
  auto starts = knots.slice(0, 0, -1);
  auto widths = knots.diff();

  DimData d;
  auto u =
    ((x.unsqueeze(0) + 1.0) * widths.unsqueeze(1) / 2.0 + starts.unsqueeze(1))
      .flatten();
  d.weights = (widths.unsqueeze(1) * w.unsqueeze(0)).flatten();
  auto spans = basis.find_spans(u);
  d.basis = basis.evaluate(u, spans, 1).to(opts);
  d.first = spans - p;
  d.num_active = p + 1;
  return d;
}

/// @brief Tensor-product basis data over a set of parametric dimensions.
struct ProductData {
  /// (Q, A) values R_a(q) (rational if the patch is).
  torch::Tensor R;
  /// (Q, A, K) parametric derivatives dR_a/dxi_k for the K dimensions.
  torch::Tensor dR;
  /// (Q, A) flat index (into `grid_shape`) of each active function.
  torch::Tensor index;
  /// (Q) product quadrature measure.
  torch::Tensor weights;
};

/// @brief Build tensor-product values over `dims`, indexing into a control
/// grid of shape `grid_shape` (one entry per dim in `dims`).
ProductData make_product(const std::vector<const DimData*>& dims,
  const std::vector<int64_t>& grid_shape, const torch::Tensor& grid_weights)
{
  const int64_t K = static_cast<int64_t>(dims.size());
  std::vector<int64_t> full_shape;
  for (auto* d : dims)
    full_shape.push_back(d->basis.size(0));
  for (auto* d : dims)
    full_shape.push_back(d->num_active);

  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  auto shaped = [&](const torch::Tensor& t, int64_t k, bool is_active) {
    // t is (n_k) or (n_k, A_k) -> broadcastable over full_shape
    std::vector<int64_t> s(2 * K, 1);
    s[k] = full_shape[k];
    if (is_active)
      s[K + k] = full_shape[K + k];
    return t.reshape(s);
  };

  auto N = torch::ones(full_shape, opts);
  std::vector<torch::Tensor> dN(K, torch::ones(full_shape, opts));
  auto weights = torch::ones(
    std::vector<int64_t>(full_shape.begin(), full_shape.begin() + K), opts);
  auto index = torch::zeros(full_shape, torch::kInt64);

  int64_t stride = 1;
  std::vector<int64_t> strides(K);
  for (int64_t k = K - 1; k >= 0; k--) {
    strides[k] = stride;
    stride *= grid_shape[k];
  }

  for (int64_t k = 0; k < K; k++) {
    const auto& d = *dims[k];
    auto val = shaped(d.basis.select(1, 0), k, true);
    auto der = shaped(d.basis.select(1, 1), k, true);
    N = N * val;
    for (int64_t j = 0; j < K; j++) {
      dN[j] = dN[j] * (j == k ? der : val);
    }
    std::vector<int64_t> ws(K, 1);
    ws[k] = full_shape[k];
    weights = weights * d.weights.reshape(ws);

    auto local = d.first.unsqueeze(1) +
                 torch::arange(d.num_active, torch::kInt64).unsqueeze(0);
    index = index + shaped(local, k, true) * strides[k];
  }

  int64_t Q = 1, A = 1;
  for (int64_t k = 0; k < K; k++) {
    Q *= full_shape[k];
    A *= full_shape[K + k];
  }
  N = N.reshape({Q, A});
  auto dNs = torch::stack(dN, -1).reshape({Q, A, K});
  index = index.expand(full_shape).reshape({Q, A}).contiguous();

  ProductData out;
  out.weights = weights.reshape({Q});
  out.index = index;
  if (grid_weights.defined()) {
    // NURBS quotient rule
    auto w = grid_weights.flatten().index({index});
    auto W = (w * N).sum(1, true);
    auto dW = (w.unsqueeze(-1) * dNs).sum(1, true);
    out.R = w * N / W;
    out.dR =
      (w.unsqueeze(-1) * dNs * W.unsqueeze(-1) - (w * N).unsqueeze(-1) * dW) /
      W.square().unsqueeze(-1);
  } else {
    out.R = N;
    out.dR = dNs;
  }
  return out;
}

} // namespace

namespace ttnte::sweep {

// =================================================================
// FaceGeometry

torch::Tensor FaceGeometry::upwind_weights(
  const torch::Tensor& omega, bool outflow) const
{
  auto dot = torch::matmul(omega, normal.transpose(0, 1));
  return torch::clamp(outflow ? dot : -dot, 0.0) * weights.unsqueeze(0);
}

// =================================================================
// PatchDiscretization

PatchDiscretization::PatchDiscretization(const cad::Patch& patch)
{
  const std::string ctx = "ttnte::sweep::PatchDiscretization";
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);

  const auto& bases = patch.get_basis();
  ndim_ = static_cast<int64_t>(bases.size());
  // get_ctrlpts_sizes() ends with the coordinate axis; keep the grid only
  auto sizes = patch.get_ctrlpts_sizes();
  shape_.assign(sizes.begin(), sizes.begin() + ndim_);
  num_dofs_ = std::accumulate(
    shape_.begin(), shape_.end(), int64_t {1}, std::multiplies<int64_t>());

  auto ctrlpts = patch.get_ctrlpts().to(opts).reshape({num_dofs_, -1});
  const int64_t phys = ctrlpts.size(1);
  if (phys != ndim_) {
    throw utils::runtime_error(ctx,
      "Sweeps require patches whose parametric and physical dimensions agree "
      "(got " +
        std::to_string(ndim_) + " parametric, " + std::to_string(phys) +
        " physical)");
  }
  torch::Tensor grid_weights;
  if (patch.is_rational()) {
    grid_weights = patch.get_weights().to(opts).contiguous();
  }

  std::vector<DimData> dims;
  dims.reserve(ndim_);
  for (const auto& b : bases) {
    dims.push_back(make_dim_data(b));
  }

  // ===============================================================
  // Volume integrals
  std::vector<const DimData*> all_dims;
  for (auto& d : dims)
    all_dims.push_back(&d);
  auto vol = make_product(all_dims, shape_, grid_weights);
  const int64_t Q = vol.R.size(0);
  const int64_t A = vol.R.size(1);

  // Geometry: J[q, i, k] = dx_i / dxi_k
  auto P_loc = ctrlpts.index({vol.index}); // (Q, A, phys)
  auto J = torch::einsum("qak,qai->qik", {vol.dR, P_loc});
  auto detJ = torch::linalg_det(J);
  if ((detJ * detJ.sign().select(0, 0) <= 0).any().item<bool>()) {
    throw utils::runtime_error(
      ctx, "The patch Jacobian changes sign or vanishes at a quadrature point");
  }
  auto Jinv = torch::linalg_inv(J); // (Q, k, i) = dxi_k / dx_i
  auto grad = torch::einsum("qak,qki->qai", {vol.dR, Jinv}); // (Q, A, ndim)
  auto Wq = detJ.abs() * vol.weights;                        // (Q)

  // Pair list over (q, a, b)
  auto rows = vol.index.unsqueeze(2).expand({Q, A, A}).reshape({-1});
  auto cols = vol.index.unsqueeze(1).expand({Q, A, A}).reshape({-1});
  auto WR = Wq.unsqueeze(1) * vol.R; // (Q, A)
  auto mass = (WR.unsqueeze(2) * vol.R.unsqueeze(1)).reshape({-1});
  std::vector<torch::Tensor> grads;
  for (int64_t i = 0; i < ndim_; i++) {
    auto Wg = Wq.unsqueeze(1) * grad.select(2, i);
    grads.push_back((Wg.unsqueeze(2) * vol.R.unsqueeze(1)).reshape({-1}));
  }

  // Coalesce duplicate (row, col) pairs
  auto keys = rows * num_dofs_ + cols;
  auto [ukeys, inverse] = at::_unique(keys, /*sorted=*/true,
    /*return_inverse=*/true);
  const int64_t P = ukeys.size(0);
  pair_row_ = torch::div(ukeys, num_dofs_, "floor");
  pair_col_ = ukeys - pair_row_ * num_dofs_;
  mass_val_ = torch::zeros({P}, opts).index_add_(0, inverse, mass);
  grad_val_ = torch::zeros({ndim_, P}, opts);
  for (int64_t i = 0; i < ndim_; i++) {
    grad_val_[i].index_add_(0, inverse, grads[i]);
  }
  basis_integral_ = torch::zeros({num_dofs_}, opts)
                      .index_add_(0, vol.index.reshape({-1}), WR.reshape({-1}));

  // ===============================================================
  // Faces
  const double orientation = patch.get_orientation();
  faces_.resize(2 * ndim_);
  for (int64_t dim = 0; dim < ndim_; dim++) {
    for (bool is_upper : {false, true}) {
      FaceGeometry& face = faces_[2 * dim + (is_upper ? 1 : 0)];
      face.dim = dim;
      face.is_upper = is_upper;
      const int64_t layer = is_upper ? shape_[dim] - 1 : 0;

      // Volume index of every face dof, C order over the remaining dims
      auto grid = torch::arange(num_dofs_, torch::kInt64).reshape(shape_);
      face.dofs = grid.select(dim, layer).flatten().contiguous();
      for (int64_t d = 0; d < ndim_; d++)
        if (d != dim)
          face.shape.push_back(shape_[d]);

      double sign = orientation * (is_upper ? 1.0 : -1.0);
      if (ndim_ == 1) {
        face.normal = torch::full({1, 1}, sign, opts);
        face.weights = torch::ones({1}, opts);
        face.pair_q = torch::zeros({1}, torch::kInt64);
        face.pair_a = torch::zeros({1}, torch::kInt64);
        face.pair_b = torch::zeros({1}, torch::kInt64);
        face.pair_val = torch::ones({1}, opts);
        face.area = 1.0;
      } else {
        std::vector<const DimData*> tdims;
        for (int64_t d = 0; d < ndim_; d++)
          if (d != dim)
            tdims.push_back(&dims[d]);
        torch::Tensor face_weights;
        if (grid_weights.defined())
          face_weights = grid_weights.select(dim, layer).contiguous();
        auto fp = make_product(tdims, face.shape, face_weights);
        const int64_t Qf = fp.R.size(0);
        const int64_t Af = fp.R.size(1);

        // Tangents along each tangential parameter
        auto Pf = ctrlpts.index({face.dofs}).index({fp.index}); // (Qf,Af,phys)
        auto T = torch::einsum("qak,qai->qki", {fp.dR, Pf});    // (Qf,K,phys)

        // Scaled outward normal, as assemble_boundary_geometry()
        const double orient = sign * ((dim % 2 != 0) ? -1.0 : 1.0);
        if (ndim_ == 2) {
          auto t = T.select(1, 0);
          face.normal =
            orient * torch::stack({t.select(1, 1), -t.select(1, 0)}, 1);
        } else {
          face.normal =
            orient * torch::linalg_cross(T.select(1, 0), T.select(1, 1), -1);
        }
        face.weights = fp.weights;
        face.area =
          (face.normal.norm(2, 1) * face.weights).sum().item<double>();

        face.pair_q = torch::arange(Qf, torch::kInt64)
                        .view({Qf, 1, 1})
                        .expand({Qf, Af, Af})
                        .reshape({-1});
        face.pair_a = fp.index.unsqueeze(2).expand({Qf, Af, Af}).reshape({-1});
        face.pair_b = fp.index.unsqueeze(1).expand({Qf, Af, Af}).reshape({-1});
        // The quadrature weight lives in upwind_weights(), not here
        face.pair_val = (fp.R.unsqueeze(2) * fp.R.unsqueeze(1)).reshape({-1});
      }

      // Locate each face pair in the coalesced volume pair list
      auto fkeys = face.dofs.index({face.pair_a}) * num_dofs_ +
                   face.dofs.index({face.pair_b});
      auto pos = torch::searchsorted(ukeys, fkeys);
      if (!(ukeys.index({pos.clamp_max(P - 1)}) == fkeys).all().item<bool>()) {
        throw utils::runtime_error(ctx,
          "A face pair has no matching volume pair (unexpected basis "
          "support on the face)");
      }
      face.pair_volume_index = pos;
    }
  }

  // ===============================================================
  // Band structure
  auto diff = pair_row_ - pair_col_;
  kl_ = std::max<int64_t>(0, diff.max().item<int64_t>());
  ku_ = std::max<int64_t>(0, (-diff).max().item<int64_t>());
  const int64_t ldab = linalg::banded::factor_ldab(kl_, ku_);
  band_index_ = pair_col_ * ldab + (kl_ + ku_ + pair_row_ - pair_col_);
}

torch::Tensor PatchDiscretization::streaming_values(
  const torch::Tensor& omega) const
{
  auto values = -torch::matmul(omega, grad_val_); // (D, P)
  for (const auto& face : faces_) {
    auto c = face.upwind_weights(omega, true); // (D, Qf)
    auto contrib = c.index({Slice(), face.pair_q}) * face.pair_val;
    values.index_add_(1, face.pair_volume_index, contrib);
  }
  return values;
}

torch::Tensor PatchDiscretization::to_band(const torch::Tensor& values) const
{
  const int64_t B = values.size(0);
  const int64_t ldab = linalg::banded::factor_ldab(kl_, ku_);
  auto out = torch::zeros({B, num_dofs_ * ldab}, values.options());
  out.index_copy_(1, band_index_, values);
  return out.view({B, num_dofs_, ldab});
}

torch::Tensor PatchDiscretization::to_dense(const torch::Tensor& values) const
{
  const int64_t B = values.size(0);
  auto out = torch::zeros({B, num_dofs_ * num_dofs_}, values.options());
  out.index_copy_(1, pair_row_ * num_dofs_ + pair_col_, values);
  return out.view({B, num_dofs_, num_dofs_});
}

torch::Tensor PatchDiscretization::apply_mass(const torch::Tensor& x) const
{
  auto y = torch::zeros_like(x);
  y.index_add_(1, pair_row_, x.index_select(1, pair_col_) * mass_val_);
  return y;
}

torch::Tensor PatchDiscretization::local_operator(
  const torch::Tensor& omega, double sigma_t) const
{
  auto values = streaming_values(omega.to(torch::kFloat64).reshape({1, -1}));
  values += sigma_t * mass_val_;
  return to_dense(values)[0];
}

} // namespace ttnte::sweep
