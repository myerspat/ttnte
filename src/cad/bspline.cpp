#include <stdexcept>

#include "ttnte/cad/basis_functions.hpp"
#include "ttnte/cad/bspline.hpp"

namespace ttnte::cad {
// Constructors
Bspline::Bspline() : BasisFunctions(), ctrl_pts_ {} {}

Bspline::Bspline(std::vector<torch::Tensor> knots, std::vector<int64_t> degrees,
  std::vector<torch::Tensor> ctrl_pts)
  : BasisFunctions(knots, degrees), ctrl_pts_(std::move(ctrl_pts))
{}

void Bspline::knot_refine(int64_t param_idx, const torch::Tensor& new_knots,
  const torch::Tensor& new_ctrl_pts)
{
  auto knots_p = knots_[param_idx];
  auto ctrl_pts_p = ctrl_pts_[param_idx];

  // check new_knots
  validate_knots(new_knots);

  // define indexing variables
  int64_t p = degree[param_idx].accessor<double, 1>();
  int64_t n = knots_p.numel();
  int64_t r = new_knots.numel();
  auto knots_old = knots_p.accessor<double, 1>();
  auto knots_new = new_knots.accessor<double, 1>();
  auto ctrl_pts_old = knots_p.accessor<double, 1>();
  auto ctrl_pts_new = new_ctrl_pts.accessor<double, 1>();
  int64_t m = n + p;
  int64_t a = find_spans(param_idx, new_coords[0]).item<int64_t>();
  int64_t b = find_spans(param_idx, new_coords[-1]).item<int64_t>() + 1;

  // Create new knot vector
  torch::Tensor final_knots = torch::empty({n + r}, torch::kFloat64);
  auto final_knots = final_knots.accessor<double, 1>();

  // Create new ctrl_pts vector
  torch::Tensor final_ctrl_pts = torch::empty({n + r}, torch::kFloat64);
  auto final_ctrl_pts = final_ctrl_pts.accessor<double, 1>();

  // Fill new knot vector
  for (int64_t j = 0; j <= a; ++j)
    fin_knots[j] = knots_old[j];
  for (int64_t j = b + p; j <= m; ++j)
    fin_knots[j + r] = knots_old[j];
  int64_t i = b + p - 1;
  int64_t k = b + p + r - 1;
  for (int64_t j = r; j >= 0; j--) {
    while (
      new_knots[j].item<int64_t>() <= knots_p[j].item<int64_t>() && i > a) {
      final_ctrl_pts[k - p - 1] = ctrl_pts_old[i - p - 1] final_knots[k] =
        knots_p[i];
      k = k - 1;
      i = i - 1;
    }
    final_ctrl_pts[k - p - 1] =
      final_ctrl_pts[k - p] for (int64_t l = 1; l <= p; l++)
    {
      ind = k - p + 1;
      alfa = final_knots[k + l] - knots_new[j];
      if (std::abs(alfa) == 0.0) {
        final_ctrl_pts[ind - 1] = final_ctrl_pts[ind];
      } else {
        alfa = alfa / (final_knots[k + 1] - knots_old[i - p + 1]);
        final_ctrl_pts[ind - 1] =
          alfa * final_ctrl_pts[ind - 1] + (1.0 - alfa) * final_ctrl_pts[ind];
      }
    }
    final_knots[k] = knots_new[j];
    k = k - 1;
  }
}

} // namespace ttnte::cad
