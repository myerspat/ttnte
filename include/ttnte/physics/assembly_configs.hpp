#pragma once

#include "ttnte/linalg/format_type.hpp"
#include <optional>
#include <torch/extension.h>

namespace ttnte::physics {

using linalg::FormatType;

/// @brief Configuration common among TT methods.
struct TTConfig {
  /// Truncation tolerance
  double eps = 1e-12;
  /// Max rank.
  int max_rank = 500;

  // =================================================================
  // Constructors
  TTConfig(double eps = 1e-12, int max_rank = 500)
    : eps(eps), max_rank(max_rank)
  {}
};

/// @brief Configuration for TT-cross.
struct CrossConfig : public TTConfig {
  /// Maximum number of sweeps.
  int nswp = 50;
  /// Rank increase each sweep.
  int kick = 4;
  /// Print progress to terminal.
  bool verbose = false;

  // =================================================================
  // Constructors
  CrossConfig(double eps = 1e-12, int max_rank = 500, int nswp = 50,
    int kick = 4, bool verbose = false)
    : TTConfig(eps, max_rank), nswp(nswp), kick(kick), verbose(verbose)
  {}
};

/// @brief General discontinuous Galerkin assembly configuration.
struct DGAssemblerConfig {
  /// Device to assemble on.
  std::optional<torch::Device> device = std::nullopt;
  /// Data type.
  std::optional<torch::ScalarType> dtype = std::nullopt;

  // =================================================================
  // Algorithm switches
  /// Maximum size to compute operations in a dense format over TT-cross.
  int64_t max_dense_size = 100000;
  /// Use TT-cross for computing the inverse of the Jacobian.
  bool cross_jacobian_inverse = false;
  /// Regularization parameter for approximating absolute value kinks
  /// in TT-cross.
  double regularization_ = 1e-6;

  // =================================================================
  // Algorithm configs
  /// TT-rounding configuration.
  TTConfig rounding;
  /// TT-cross interpolation configuration.
  CrossConfig cross;

  // =================================================================
  // Constructors
  DGAssemblerConfig(double eps = 1e-12, int max_rank = 500, int nswp = 50,
    int kick = 4, int64_t max_dense_size = 1e5,
    bool cross_jacobian_inverse = false,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt, bool verbose = false)
    : device(device), dtype(dtype), max_dense_size(max_dense_size),
      cross_jacobian_inverse(cross_jacobian_inverse), rounding(eps, max_rank),
      cross(eps, max_rank, nswp, kick, verbose)
  {}
  DGAssemblerConfig(const TTConfig& rounding, const CrossConfig& cross,
    int64_t max_dense_size = 1e5, bool cross_jacobian_inverse = false,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::ScalarType> dtype = std::nullopt)
    : device(device), dtype(dtype), max_dense_size(max_dense_size),
      cross_jacobian_inverse(cross_jacobian_inverse), rounding(rounding),
      cross(cross)
  {}
};

/// @brief Assembly configuration for transport specifically.
struct DGTransportAssemblerConfig : public DGAssemblerConfig {
  /// Format of the interior loss operator (streaming + collision).
  FormatType interior_loss_fmt = FormatType::TENSOR_TRAIN;
  /// Format of the scattering operator.
  FormatType scatter_fmt = FormatType::TENSOR_TRAIN;
  /// Format of the fission operator.
  FormatType fission_fmt = FormatType::TENSOR_TRAIN;
  /// Outflow boundary operator format.
  FormatType outflow_fmt = FormatType::TENSOR_TRAIN;
  /// Inflow boundary operator format.
  FormatType inflow_fmt = FormatType::TENSOR_TRAIN;
  /// Format for the source vector.
  FormatType source_fmt = FormatType::TENSOR_TRAIN;
};

} // namespace ttnte::physics
