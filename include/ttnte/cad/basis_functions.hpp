#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <optional>

namespace ttnte::cad {
class Basis_Functions : public std::enable_shared_from_this<Basis_Functions>{
public:
    // Constructors
    Basis_Functions();

    Basis_Functions( std::vector<torch::Tensor> knots,
          std::vector<int64_t> degree);

    // Basic B-Spline data (in k-dimension physical space)
    std::vector<torch::Tensor> knots;  ///< vector of knots
    std::vector<int64_t> degree;  ///< degrees in each parametric dim

private:

}
} // namespace ttnte::cad
