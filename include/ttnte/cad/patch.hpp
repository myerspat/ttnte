#pragma once

#include "ttnte/cad/bspline_basis.hpp"
#include "ttnte/utils/label.hpp"
#include <ATen/Dispatch.h>
#include <optional>
#include <string>

namespace ttnte::cad {

// TODO: Methods to implement
// [ ] Inverse spline evaluation
// [ ] Basis function method (list of coordinates)
// [ ] Quadrature basis function method
// [ ] Gradient Basis function method (list of coordinates)
// [ ] Quadrature gradient basis function method
// [ ] Jacobian method (list of coordinates)
// [ ] Jacobian method on quadrature (out source to torchTT for TT)
// [ ] Normal evaluation (list of coordinates)
// [ ] Quadrature normal evaluate
// [ ] Initialize boundary method
// [ ] Boundary orientation method
// [ ] Load from hdf5 method
// [ ] Save to hdf5 method
// [ ] Knot insertion
// [ ] Order elevation
// [ ] A general refine method that uses knot insertion / order elevation
// [ ] Split patch method
// [ ] Functions related to solution fields (support both tensors and TTs)
// [ ] Functions related to MPI communication

class Patch {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<Patch>;
  using Basis = c10::SmallVector<BSplineBasis, 3>;

private:
  // =================================================================
  // Private data
  Label label_;
  Basis basis_;
  bool is_valid_ = false;
  bool is_rational_ = false;

  // B-Spline: stores (N_1, ..., N_D, dim)
  // NURBS: stores (N_1, ..., N_D, dim + 1) where the last dim is (wx, wy, wz)
  torch::Tensor ctrlptsw_;

  // =================================================================
  // Private constructors
  Patch(const torch::Tensor& ctrlptsw, const Basis& basis, bool is_rational,
    bool is_valid, const Label& label)
    : ctrlptsw_(ctrlptsw), basis_(basis), is_rational_(is_rational),
      is_valid_(is_valid), label_(label)
  {}

  // =================================================================
  // Private methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return std::string("ttnte::cad::Patch::") + func_name;
  }

public:
  // =================================================================
  // Public constructors
  Patch(std::optional<std::string> label = std::nullopt);

  Patch(const torch::Tensor& ctrlpts, const Basis& basis,
    bool is_rational = false, std::optional<std::string> label = std::nullopt);

  Patch(const torch::Tensor& ctrlpts, const torch::Tensor& weights,
    const Basis& basis, std::optional<std::string> label = std::nullopt);

  // =================================================================
  // Public methods
  void validate();
  void invalidate() { is_valid_ = false; }
  bool is_valid() const noexcept { return is_valid_; }
  bool is_rational() const noexcept { return is_rational_; }
  Patch clone() const;

  torch::Tensor evaluate(const torch::Tensor& local_coords);
  torch::Tensor evaluate(
    const c10::SmallVector<torch::Tensor, 3>& local_coords);

  // =================================================================
  // Public Getters / Setters
  const inline Label& get_label() const noexcept { return label_; }
  const inline torch::Tensor& get_ctrlptsw() const noexcept
  {
    return ctrlptsw_;
  }
  const inline Basis& get_basis() const noexcept { return basis_; }
  inline int64_t get_ndim() const noexcept { return basis_.size(); }
  torch::Tensor get_ctrlpts() const;
  torch::Tensor get_weights() const;
  int64_t get_order(size_t dim) const;
  std::vector<int64_t> get_orders() const;
  int64_t get_degree(size_t dim) const;
  std::vector<int64_t> get_degrees() const;
  int64_t get_ctrlpts_size(size_t dim) const;
  std::vector<int64_t> get_ctrlpts_sizes() const;
  torch::Device get_device();
  torch::ScalarType get_dtype();

  torch::Tensor get_bbox(double epsilon = 0.0) const;
  Patch get_boundary(size_t dim, bool is_upper, bool clone = true);

  void inline set_label(const Label& label) { label_ = label; }
  void inline set_label(const std::string& label)
  {
    label_ = Label::from_string(label);
  }
  void set_ctrlptsw(const torch::Tensor& ctrlptsw, bool clone = true);
  void set_basis(const Basis& basis, bool clone = true);
  void set_ctrlpts(torch::Tensor ctrlpts, bool clone = true);
  void set_weights(torch::Tensor weights, bool clone = true);
};

std::ostream& operator<<(std::ostream& os, const Patch& p);

} // namespace ttnte::cad
