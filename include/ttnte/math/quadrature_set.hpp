#pragma once

#include "ttnte/utils/exception.hpp"
#include <c10/util/SmallVector.h>
#include <cstdint>
#include <memory>
#include <numbers>
#include <torch/extension.h>

namespace ttnte::math {

enum class Symmetry : uint8_t { NONE, XY_PLANE };

/// @brief Quadrature set base class.
class QuadratureSet {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<QuadratureSet>;

protected:
  // =================================================================
  // Protected data
  /// The weighting factor applied to the weights.
  double weighting_factor;
  /// Whether the quadrature set is a tensor product.
  bool is_tensor_product_;
  /// Any symmetry for this quadrature set.
  Symmetry symmetry_;

  // =================================================================
  // Protected constructors
  QuadratureSet(double weighting_factor, bool is_tensor_product,
    Symmetry symmetry = Symmetry::NONE)
    : weighting_factor(weighting_factor), is_tensor_product_(is_tensor_product),
      symmetry_(symmetry)
  {}

public:
  virtual ~QuadratureSet() = default;

  // =================================================================
  // Public methods
  /// @return Is the quadrature a tensor product of multple dimensions?
  bool is_tensor_product() const noexcept { return is_tensor_product_; }

  /// @brief In-place send the data of this quadrature set to a device or to
  /// another data type.
  /// @param options The tensor options.
  virtual void to_(const torch::TensorOptions& options) = 0;

  /// @brief In-place send the data of this quadrature set to a device or to
  /// another data type.
  void to_(const torch::Device& device, const torch::ScalarType& dtype);
  void to_(const torch::ScalarType& dtype);
  void to_(const torch::Device& device);

  // =================================================================
  // Public Getters / Setters
  /// @return The quadrature points.
  virtual torch::Tensor get_points() const = 0;
  /// @return The weights of the quadrature points (note these sum to one).
  virtual torch::Tensor get_weights() const = 0;
  /// @return Get the number of quadrature points.
  virtual int64_t get_num_dofs() const = 0;
  /// @return The current device of the quadrature set.
  virtual torch::Device get_device() const = 0;
  /// @return the current data type of the quadrature set.
  virtual torch::ScalarType get_dtype() const = 0;
  /// @return The weighting factor to apply to the quadrature weights.
  double get_weighting_factor() const noexcept { return weighting_factor; }
  /// @return The symmetry of this quadrature set.
  Symmetry get_symmetry() const noexcept { return symmetry_; }
};

/// @brief A 1-D quadrature.
class QuadratureSet1D : public QuadratureSet {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<QuadratureSet1D>;

protected:
  // =================================================================
  // Protected data
  /// The quadrature points in the range (-1, 1).
  torch::Tensor points_;
  /// The quadrature weights which sum to 1.
  torch::Tensor weights_;

  // =================================================================
  // Protected constructors
  QuadratureSet1D(const torch::Tensor& points, const torch::Tensor& weights,
    double weighting_factor, Symmetry symmetry = Symmetry::NONE)
    : QuadratureSet(weighting_factor, false, symmetry), points_(points),
      weights_(weights)
  {
    assert(points_.ndimension() == weights_.ndimension() &&
           points_.size(0) == weights_.size(0) &&
           points_.device() == weights_.device() &&
           points_.dtype() == weights_.dtype());
  }

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new QuadratureSet1D(std::forward<Args>(args)...));
  }

  /// @brief Compute a degree n Gauss-Legendre quadrature.
  /// @param n The degree of the quadrature set.
  /// @param weighting_factor The weighting factor applied to the weights.
  /// @return A pointer to the new 1-D quadrature.
  static Ptr gauss_legendre(int64_t n, double weighting_factor = 2.0);
  /// @brief Compute a Gauss-Chebyshev quadrature.
  /// @param n The degree of the quadrature set.
  /// @param weighting_factor The weighting factor applied to the weights.
  /// @return A pointer to the new 1-D quadrature.
  static Ptr gauss_chebyshev(
    int64_t n, double weighting_factor = 2.0 * std::numbers::pi);

  /// @brief In-place send the data of this quadrature set to a device or to
  /// another data type.
  /// @param options The tensor options.
  void to_(const torch::TensorOptions& options) override final;

  // =================================================================
  // Public Getters / Setters
  /// @return The quadrature points.
  torch::Tensor get_points() const final override { return points_; }
  /// @return The weights of the quadrature points (note these sum to one).
  torch::Tensor get_weights() const final override { return weights_; }
  /// @return Get the number of quadrature points.
  int64_t get_num_dofs() const final override { return points_.size(0); }
  /// @return The current device of the quadrature set.
  torch::Device get_device() const final override { return points_.device(); };
  /// @return the current data type of the quadrature set.
  torch::ScalarType get_dtype() const final override
  {
    return points_.dtype().toScalarType();
  }
};

/// @brief A tensor product quadrature for integrating the unit sphere.
class ProductQuadrature : public QuadratureSet {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<ProductQuadrature>;
  using Quads = c10::SmallVector<QuadratureSet1D::Ptr, 3>;

protected:
  // =================================================================
  // Protected data
  /// Quadrature sets along each dimension
  Quads quads_;

  // =================================================================
  // Protected constructors
  ProductQuadrature(const Quads& quads, double weighting_factor = 1.0,
    Symmetry symmetry = Symmetry::NONE)
    : QuadratureSet(weighting_factor, true, symmetry), quads_(quads)
  {
    if (quads_.empty()) {
      throw utils::runtime_error(
        "ttnte::physics::ProductQuadrature::ProductQuadrature",
        "`quads` cannot be empty");
    }
  }

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new ProductQuadrature(std::forward<Args>(args)...));
  }

  /// @brief Compute a Chebyshev-Legendre quadrature.
  /// @param n The order of the quadrature set.
  /// @param ndim The number of dimensions (either 2 or 3). If there are 2
  /// spatial dimensions then we use symmetry across the xy-plane to half the
  /// number of ordinates.
  /// @param weighting_factor The weighting factor applied to the weights.
  /// @return A pointer to the new tensor product quadrature.
  static Ptr gauss_legendre_chebyshev(int64_t n_polar, int64_t n_azimuthal,
    int64_t ndim = 3,
    double weighting_factor = static_cast<double>(4.0) * std::numbers::pi);

  /// @brief In-place send the data of this quadrature set to a device or to
  /// another data type.
  /// @param options The tensor options.
  void to_(const torch::TensorOptions& options) override final;

  // =================================================================
  // Public Getters / Setters
  /// @return The quadrature sets along each dimension.
  const Quads& get_quads() const noexcept { return quads_; }
  /// @return The factored tensor product points.
  c10::SmallVector<torch::Tensor, 3> get_factored_points() const noexcept;
  /// @return The factored tensor product weights.
  c10::SmallVector<torch::Tensor, 3> get_factored_weights() const noexcept;
  /// @return The number of dimensions (quadratures).
  int64_t get_ndim() const noexcept { return quads_.size(); }
  /// @return The quadrature points.
  torch::Tensor get_points() const final override;
  /// @return The weights of the quadrature points (note these sum to one).
  torch::Tensor get_weights() const final override;
  /// @return Get the number of quadrature points.
  int64_t get_num_dofs() const final override;
  /// @return The current device of the quadrature set.
  torch::Device get_device() const final override
  {
    return quads_[0]->get_device();
  };
  /// @return the current data type of the quadrature set.
  torch::ScalarType get_dtype() const final override
  {
    return quads_[0]->get_dtype();
  }
};

} // namespace ttnte::math
