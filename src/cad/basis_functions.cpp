#include "ttnte/cad/basis_functions.hpp"

namespace ttnte::cad {
// Constructors
BasisFunctions::BasisFunctions():degree(0)
{
}

BasisFunctions::BasisFunctions( std::vector<torch::Tensor> knots,
  std::vector<int64_t> degree): knots(knots), 
  degree(degree)
{
}


torch::Tensor find_spans(int64_t param_idx, torch::Tensor coords)
{
}

// Single Dimension Evaluation
torch::Tensor basis_functions_bspline(
        int64_t param_idx,
        torch::Tensor spans,
        torch::Tensor coords)
{
  auto knots = knots[param_idx];
  int64_t p = degree[param_idx];
    
  int64_t num_coords = coords.size(0);
  auto basis = torch::zeros({num_coords, p + 1}, torch::kFloat64);
    
  auto knots_a = knots.accessor<double, 1>();
  auto spans_a = spans.accessor<int64_t, 1>();
  auto coords_a = coords.accessor<double, 1>();
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
        
torch::Tensor basis_functions_ders_bspline(
        int64_t param_idx,
        torch::Tensor spans,
        torch::Tensor coords, 
        int64_t order)
{}

}