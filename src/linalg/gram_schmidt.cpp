#include "ttnte/linalg/gram_schmidt.hpp"
#include "ttnte/utils/exception.hpp"
#include <cmath>

namespace ttnte::linalg {

MGS2Result mgs2a(
  const torch::Tensor& Q, torch::Tensor& x, const MGS2Options& options)
{
  const char* where = "ttnte::linalg::mgs2a";
  if (Q.dim() != 2 || x.dim() != 1 || Q.size(1) != x.size(0))
    throw utils::runtime_error(where, "Expected Q (r, n) and x (n)");
  if (Q.scalar_type() != x.scalar_type() || Q.device() != x.device())
    throw utils::runtime_error(where, "Q and x differ in dtype or device");
  if (!x.is_contiguous())
    throw utils::runtime_error(where, "x must be contiguous");
  if (options.max_projections < 2)
    throw utils::runtime_error(where, "max_projections must be >= 2");
  if (!(options.eps > 0.0))
    throw utils::runtime_error(where, "eps must be positive");
  if (!(options.rtol >= 0.0))
    throw utils::runtime_error(where, "rtol must be >= 0");

  // Invariant: x_in = Q^T c + alpha u, with u = x / alpha after normalizing
  const int64_t r = Q.size(0), n = x.size(0);
  MGS2Result out;
  out.norm = x.norm().item<double>();
  auto c = torch::zeros({r}, x.options());
  auto project = [&] {
    const auto dc = Q.mv(x);
    x.addmv_(Q.t(), dc, 1.0, -1.0);
    c += dc;
  };
  // Not a new direction: exactly zero, in the span to rtol, or Q full
  auto in_span = [&] {
    return out.alpha == 0.0 || out.alpha <= options.rtol * out.norm || r >= n;
  };
  auto finish = [&](bool appended) {
    out.coefficients = c;
    out.appended = appended;
    return out;
  };

  if (r > 0) {
    project();
    project();
  }
  out.alpha = x.norm().item<double>();
  if (in_span())
    return finish(false);
  x /= out.alpha;
  if (r > 0) {
    auto first = [&] { return std::abs(Q[0].dot(x).item<double>()); };
    for (int64_t m = 0; m < options.max_projections - 2; m++) {
      if (first() <= options.eps)
        break;
      x *= out.alpha;
      project();
      out.alpha = x.norm().item<double>();
      if (in_span())
        return finish(false);
      x /= out.alpha;
    }
    out.orthogonal = first() <= options.eps;
  }
  return finish(true);
}

} // namespace ttnte::linalg
