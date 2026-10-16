#include "ttnte/sweep/sum_factorization.hpp"
#include <algorithm>

namespace ttnte::sweep {

SumFactorization::SumFactorization(const PatchDiscretization& disc)
  : disc_(&disc), ndim_(disc.ndim()), num_dofs_(disc.num_dofs())
{
  const auto& bases = disc.dim_basis();
  num_points_ = 1;
  for (const auto& b : bases) {
    n_.push_back(b.num_functions);
    nq_.push_back(b.first.size(0));
    p1_.push_back(b.values.size(1));
    num_points_ *= b.first.size(0);
  }
  auto copy = [](const torch::Tensor& t) {
    auto c = t.to(torch::kFloat64).contiguous();
    return std::vector<double>(
      c.data_ptr<double>(), c.data_ptr<double>() + c.numel());
  };
  measure_ = copy(disc.quad_measure());
  inverse_jacobian_ = copy(disc.quad_inverse_jacobian());
  W_.assign(num_points_, 1.0);
  dW_.assign(num_points_ * ndim_, 0.0);
  if (disc.control_weights().defined()) {
    control_weights_ = copy(disc.control_weights());
    // W and its parametric gradient on the quadrature grid
    std::vector<int64_t> shape(n_);
    std::vector<double> a(num_dofs_), b;
    for (int64_t k = -1; k < ndim_; k++) {
      std::copy(control_weights_.begin(), control_weights_.end(), a.begin());
      shape = n_;
      for (int64_t dim = 0; dim < ndim_; dim++) {
        int64_t size = 1;
        for (int64_t j = 0; j < ndim_; j++)
          size *= j == dim ? nq_[j] : shape[j];
        b.assign(size, 0.0);
        forward(dim, dim == k, shape, a.data(), b.data());
        shape[dim] = nq_[dim];
        a.swap(b);
      }
      for (int64_t q = 0; q < num_points_; q++) {
        if (k < 0)
          W_[q] = a[q];
        else
          dW_[q * ndim_ + k] = a[q];
      }
    }
  }
}

void SumFactorization::forward(int64_t dim, bool derivative,
  const std::vector<int64_t>& shape, const double* in, double* out) const
{
  const auto& basis = disc_->dim_basis()[dim];
  const int64_t* first = basis.first.data_ptr<int64_t>();
  const double* M =
    (derivative ? basis.derivs : basis.values).data_ptr<double>();
  const int64_t p1 = p1_[dim], nq = nq_[dim], n = shape[dim];
  int64_t outer = 1, inner = 1;
  for (int64_t j = 0; j < dim; j++)
    outer *= shape[j];
  for (int64_t j = dim + 1; j < ndim_; j++)
    inner *= shape[j];
  for (int64_t o = 0; o < outer; o++)
    for (int64_t q = 0; q < nq; q++) {
      double* dst = out + (o * nq + q) * inner;
      std::fill(dst, dst + inner, 0.0);
      for (int64_t t = 0; t < p1; t++) {
        const double c = M[q * p1 + t];
        const double* src = in + (o * n + first[q] + t) * inner;
        for (int64_t i = 0; i < inner; i++)
          dst[i] += c * src[i];
      }
    }
}

void SumFactorization::transpose(int64_t dim, bool derivative,
  const std::vector<int64_t>& shape, const double* in, double* out) const
{
  const auto& basis = disc_->dim_basis()[dim];
  const int64_t* first = basis.first.data_ptr<int64_t>();
  const double* M =
    (derivative ? basis.derivs : basis.values).data_ptr<double>();
  const int64_t p1 = p1_[dim], nq = nq_[dim], n = n_[dim];
  int64_t outer = 1, inner = 1;
  for (int64_t j = 0; j < dim; j++)
    outer *= shape[j];
  for (int64_t j = dim + 1; j < ndim_; j++)
    inner *= shape[j];
  std::fill(out, out + outer * n * inner, 0.0);
  for (int64_t o = 0; o < outer; o++)
    for (int64_t q = 0; q < nq; q++) {
      const double* src = in + (o * nq + q) * inner;
      for (int64_t t = 0; t < p1; t++) {
        const double c = M[q * p1 + t];
        double* dst = out + (o * n + first[q] + t) * inner;
        for (int64_t i = 0; i < inner; i++)
          dst[i] += c * src[i];
      }
    }
}

void SumFactorization::apply(const double* omega, double sigma, const double* u,
  double* y, Workspace& work) const
{
  const int64_t K = ndim_, Q = num_points_, N = num_dofs_;
  int64_t big = std::max(N, Q);
  for (int64_t k = 0; k < K; k++) {
    int64_t s = 1;
    for (int64_t j = 0; j < K; j++)
      s *= std::max(n_[j], nq_[j]);
    big = std::max(big, s);
  }
  work.a.resize(big);
  work.b.resize(big);
  work.sum.resize(N);
  const bool rational = !control_weights_.empty();

  // u at the quadrature points: (B_1 x ... x B_K)(w u) / W
  double* a = work.a.data();
  double* b = work.b.data();
  for (int64_t v = 0; v < N; v++)
    a[v] = rational ? control_weights_[v] * u[v] : u[v];
  std::vector<int64_t> shape(n_);
  for (int64_t dim = 0; dim < K; dim++) {
    forward(dim, false, shape, a, b);
    shape[dim] = nq_[dim];
    std::swap(a, b);
  }
  // Pointwise fields: h_0 tested against values, h_k against d/dxi_k
  std::vector<double>& fields = work.face;
  fields.resize((K + 1) * Q);
  for (int64_t q = 0; q < Q; q++) {
    const double W = W_[q];
    const double uq = a[q] / W;
    const double m = measure_[q];
    double h0 = sigma * m * uq / W;
    for (int64_t k = 0; k < K; k++) {
      double c = 0.0;
      for (int64_t i = 0; i < K; i++)
        c += inverse_jacobian_[(q * K + k) * K + i] * omega[i];
      const double g = -m * c * uq;
      fields[(k + 1) * Q + q] = g / W;
      h0 -= g * dW_[q * K + k] / (W * W);
    }
    fields[q] = h0;
  }
  // Test: y = w * [(B^T x ... x B^T) h_0 + sum_k (.. D_k^T ..) h_k]
  std::fill(y, y + N, 0.0);
  for (int64_t k = -1; k < K; k++) {
    std::copy(fields.begin() + (k + 1) * Q, fields.begin() + (k + 2) * Q, a);
    shape = nq_;
    for (int64_t dim = K - 1; dim >= 0; dim--) {
      transpose(dim, dim == k, shape, a, b);
      shape[dim] = n_[dim];
      std::swap(a, b);
    }
    for (int64_t v = 0; v < N; v++)
      y[v] += a[v];
  }
  if (rational)
    for (int64_t v = 0; v < N; v++)
      y[v] *= control_weights_[v];

  // Outflow faces: <(Omega . n)_+ R_v, u>_f
  const auto& view = disc_->view();
  for (const auto& face : view.faces) {
    std::vector<double> c(face.num_points);
    bool any = false;
    for (int64_t q = 0; q < face.num_points; q++) {
      double dot = 0.0;
      for (int64_t i = 0; i < K; i++)
        dot += omega[i] * face.normal[q * K + i];
      c[q] = dot > 0.0 ? dot * face.weights[q] : 0.0;
      any |= c[q] != 0.0;
    }
    if (!any)
      continue;
    for (int64_t p = 0; p < face.num_pairs; p++)
      y[face.dofs[face.pair_a[p]]] +=
        c[face.pair_q[p]] * face.pair_val[p] * u[face.dofs[face.pair_b[p]]];
  }
}

} // namespace ttnte::sweep
