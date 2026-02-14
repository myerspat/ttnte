#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <optional>

namespace ttnte::cad {
class BasisFunctions : public std::enable_shared_from_this<BasisFunctions>{
public:
    // Constructors
    BasisFunctions();

    BasisFunctions( std::vector<torch::Tensor> knots,
          std::vector<int64_t> degree);

    // Basic B-Spline data (in k-dimension physical space)
    std::vector<torch::Tensor> knots;  ///< vector of knots 
    std::vector<int64_t> degree;  ///< degrees in each parametric dimension

    // Getters / Setters

    // Evaluation Methods
    torch::Tensor find_spans(int64_t param_idx, torch::Tensor coords);
private:
    // Private Helper Methods
    
    //Evaluate B-spline basis functions along a single parametric dimension
    torch::Tensor basis_functions_bspline(
        int64_t param_idx,
        torch::Tensor spans,
        torch::Tensor coords);

    //Evaluate B-spline basis function derivatives along a single parametric dimension
    torch::Tensor basis_functions_ders_bspline(
        int64_t param_idx,
        torch::Tensor spans,
        torch::Tensor coords,
        int64_t order); 
}
} // namespace ttnte::cad
