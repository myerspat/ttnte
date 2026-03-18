#pragma once

#include "ttnte/cad/bspline_basis.hpp"
#include "ttnte/utils/label.hpp"
#include "ttnte/utils/mpi_helpers.hpp"
#include <optional>
#include <string>
#include <torch/extension.h>

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
  uint64_t fill_id_ = 0;

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

  // MPI communication
  template<typename T>
  void inline pack(
    std::vector<int64_t>& meta_buffer, std::vector<T>& payload_buffer) const
  {
    // Fill the meta data buffer first
    meta_buffer.push_back(label_.to_int()); // Patch label
    meta_buffer.push_back(is_valid_);       // Is valid Boolean
    meta_buffer.push_back(is_rational_);    // Is rational Boolean
    meta_buffer.push_back(basis_.size());   // Number of parametric dimensions

    // Iterate through the basis
    for (const auto& b : basis_) {
      b.pack(meta_buffer, payload_buffer);
    }

    // Add the control point information
    meta_buffer.push_back(ctrlptsw_.numel()); // Number of control points
    auto ctrlptsw_c = ctrlptsw_.flatten().contiguous();
    payload_buffer.insert(payload_buffer.end(), ctrlptsw_c.data_ptr<T>(),
      ctrlptsw_c.data_ptr<T>() + ctrlptsw_c.numel());
  }

  void pack(std::vector<int64_t>& meta_buffer,
    std::vector<torch::Tensor>& payload_buffer) const;

  template<typename T>
  static inline Patch unpack(const int64_t* meta_buffer,
    const T* payload_buffer, int& meta_idx, int& payload_idx)
  {
    // Patch metadata
    Label label = Label(static_cast<uint64_t>(meta_buffer[meta_idx++]));
    bool is_valid = static_cast<bool>(meta_buffer[meta_idx++]);
    bool is_rational = static_cast<bool>(meta_buffer[meta_idx++]);
    int64_t ndim = meta_buffer[meta_idx++];

    // Build the basis
    Basis basis;
    basis.reserve(ndim);
    c10::SmallVector<int64_t, 4> shape;
    shape.reserve(ndim + 1);

    for (size_t d = 0; d < ndim; d++) {
      basis.push_back(BSplineBasis::unpack(
        meta_buffer, payload_buffer, meta_idx, payload_idx));
      shape.push_back(basis.back().get_size());
    }
    shape.push_back(static_cast<int64_t>(-1));

    // Control points
    torch::Tensor ctrlptsw =
      utils::unpack_tensor(&payload_buffer[payload_idx], std::move(shape),
        torch::TensorOptions()
          .device(torch::kCPU)
          .dtype(torch::CppTypeToScalarType<T>::value));
    payload_idx += meta_buffer[meta_idx++];

    return Patch(std::move(ctrlptsw), std::move(basis), std::move(is_rational),
      std::move(is_valid), std::move(label));
  }

  // =================================================================
  // Public Getters / Setters
  const inline Label& get_label() const noexcept { return label_; }
  const inline torch::Tensor& get_ctrlptsw() const noexcept
  {
    return ctrlptsw_;
  }
  const inline Basis& get_basis() const noexcept { return basis_; }
  const inline uint64_t& get_fill_id() const noexcept { return fill_id_; }
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

  template<typename T>
  utils::Label<T> get_fill() const noexcept
  {
    return utils::Label<T>(fill_id_);
  }

  void inline set_label(const Label& label) { label_ = label; }
  void inline set_label(const std::string& label)
  {
    label_ = Label::from_string(label);
  }
  void set_ctrlptsw(const torch::Tensor& ctrlptsw, bool clone = true);
  void set_basis(const Basis& basis, bool clone = true);
  void set_ctrlpts(torch::Tensor ctrlpts, bool clone = true);
  void set_weights(torch::Tensor weights, bool clone = true);
  void set_fill_id(const uint64_t& id) { fill_id_ = id; }

  template<typename T>
  void set_fill(const utils::Label<T>& label)
  {
    fill_id_ = label.to_int();
  }
};

std::ostream& operator<<(std::ostream& os, const Patch& p);

} // namespace ttnte::cad
