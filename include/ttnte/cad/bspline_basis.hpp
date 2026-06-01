#pragma once

#include "ttnte/cad/basis.hpp"
#include "ttnte/utils/mpi_helpers.hpp"
#include <cstdint>
#include <ostream>
#include <torch/extension.h>

namespace ttnte::cad {

class BSplineBasis : public Basis<BSplineBasis> {
public:
  // =================================================================
  // Public data
  static constexpr const char* class_name = "BSplineBasis";

private:
  // =================================================================
  // Private data
  /// 1-D tensor of increasing, possibly repeating values in the range of [0,
  /// 1].
  torch::Tensor knotvector_;
  /// The polynomial degree of the B-Splines.
  int64_t degree_;
  /// Is this BSplineBasis immutable?
  bool is_finalized_ = false;

public:
  // =================================================================
  // Public constructors
  BSplineBasis(const torch::Tensor& knotvector, int64_t degree,
    std::optional<std::string> label = std::nullopt);
  BSplineBasis(
    const torch::Tensor& knotvector, int64_t degree, const Label& label);
  BSplineBasis(const torch::Tensor& knotvector, int64_t degree,
    bool is_finalized, const Label& label);

  // =================================================================
  // Public methods
  /// @return Is this BSplineBasis immutable?
  bool is_finalized() const noexcept { return is_finalized_; }
  /// @brief Check this BSplineBasis is valid and mark it as immutable.
  /// @param knotvector_view The view into the full data tensor for the
  /// knot vector.
  void finalize(const torch::Tensor& knotvector_view);
  /// @brief Normalize the knot vector between [0, 1].
  void normalize_knotvector();
  BSplineBasis& to_(std::optional<torch::Device> device,
    std::optional<torch::ScalarType> dtype);
  BSplineBasis to(std::optional<torch::Device> device,
    std::optional<torch::ScalarType> dtype) const;
  BSplineBasis clone() const;

  // // TODO: Find the first index where the interval begins for u
  // // For example if x = 0.5 and knotvector_ = [0, 0, 0, 1, 1, 1]
  // // then we return 2
  // int64_t inline find_span(const double& u) const
  // {
  //   throw utils::runtime_error(
  //     *this, error_context("find_span"), "Not implemented yet");
  // }

  /// @brief Find the knot spans (indices into the knot vector where the knot
  /// changes from one index to the next) for a tensor of parametric
  /// coordinates.
  /// @param u A 1-D tensor of parametric coordinates between [0, 1].
  /// @return A 1-D tensor of the first index into the knot vector.
  torch::Tensor find_spans(const torch::Tensor& u) const;

  /// @brief Evaluate all the non-zero basis functions and their derivatives (if
  /// ``derivative_order > 0``) at a set of parametric coordinates.
  /// @param u A 1-D tensor of parametric coordinates between [0, 1].
  /// @param derivative_order The number of derivatives to return.
  /// @return A 2-D tensor of shape (n, m) where n is the number of values in
  /// ``u`` and m is the number of non-zero basis functions.
  torch::Tensor evaluate(
    const torch::Tensor& u, const int64_t& derivative_order = 0) const;

  /// @brief Evaluate all the non-zero basis functions and their derivatives (if
  /// ``derivative_order > 0``) at a set of parametric coordinates.
  /// @param u A 1-D tensor of parametric coordinates between [0, 1].
  /// @param spans A 1-D tensor of span indices.
  /// @param derivative_order The number of derivatives to return.
  /// @return A 2-D tensor of shape (n, m) where n is the number of values in
  /// ``u`` and m is the number of non-zero basis functions.
  torch::Tensor evaluate(const torch::Tensor& u, const torch::Tensor& spans,
    const int64_t& derivative_order = 0) const;

  /// @brief Evaluate all basis functions (zero included) to build a (n, m)
  /// tensor where n is the number of points in u and m is the number of basis
  /// functions in the BSplineBasis.
  /// @param u A 1-D tensor of parametric coordinates between [0, 1].
  /// @param derivative_order The number of derivatives to return.
  /// @return The result evaluations.
  torch::Tensor evaluate_all(
    const torch::Tensor& u, const int64_t& derivative_order = 0) const;

  /// @brief Evaluate all basis functions (zero included) to build a (n, m)
  /// tensor where n is the number of points in u and m is the number of basis
  /// functions in the BSplineBasis.
  /// @param u A 1-D tensor of parametric coordinates between [0, 1].
  /// @param spans A 1-D tensor of span indices.
  /// @param derivative_order The number of derivatives to return.
  /// @return The result evaluations.
  torch::Tensor evaluate_all(const torch::Tensor& u, const torch::Tensor& spans,
    const int64_t& derivative_order = 0) const;

  // MPI communication
  template<typename T>
  void inline pack(
    std::vector<int64_t>& meta_buffer, std::vector<T>& payload_buffer) const
  {
    // Fill the meta data buffer first
    meta_buffer.push_back(label_.to_int());     // BSplineBasis label
    meta_buffer.push_back(knotvector_.size(0)); // Size of the knot vector
    meta_buffer.push_back(degree_);             // Polynomial degree

    // Add knot vector to the payload
    auto knotvector_c = knotvector_.contiguous();
    payload_buffer.insert(payload_buffer.end(), knotvector_c.data_ptr<T>(),
      knotvector_c.data_ptr<T>() + knotvector_c.numel());
  }

  void pack(std::vector<int64_t>& meta_buffer,
    std::vector<torch::Tensor>& payload_buffer) const;

  template<typename T>
  static inline BSplineBasis unpack(const int64_t* meta_buffer,
    const T* payload_buffer, int& meta_idx, int& payload_idx)
  {
    // BSplineBasis metadata
    Label label = Label(static_cast<uint64_t>(meta_buffer[meta_idx++]));
    int64_t kv_size = meta_buffer[meta_idx++];
    int64_t degree = meta_buffer[meta_idx++];

    // Get knot vector
    torch::Tensor knotvector = utils::unpack_tensor(
      &payload_buffer[payload_idx], torch::IntArrayRef {kv_size},
      torch::TensorOptions()
        .device(torch::kCPU)
        .dtype(torch::CppTypeToScalarType<T>::value));
    payload_idx += kv_size;

    return BSplineBasis(
      std::move(knotvector), std::move(degree), std::move(label));
  }

  // =================================================================
  // Public getters
  /// @return The polynomial degree of the basis.
  const inline int64_t& get_degree() const noexcept { return degree_; }
  /// @return The knot vector of the basis.
  const inline torch::Tensor& get_knotvector() const noexcept
  {
    return knotvector_;
  }
  /// @return The order of the basis.
  inline int64_t get_order() const noexcept { return degree_ + 1; }
  /// @return A tuple of unique knots in the knot vector and the multiplicity of
  /// each.
  inline std::tuple<torch::Tensor, torch::Tensor>
  get_unique_knots_and_multiplicity() const noexcept
  {
    auto result = torch::unique_consecutive(knotvector_, false, true);
    return std::make_tuple(std::get<0>(result), std::get<2>(result));
  }
  /// @return The unique knots in the knot vector.
  inline torch::Tensor get_unique_knots() const noexcept
  {
    return std::get<0>(get_unique_knots_and_multiplicity());
  }
  /// @return The multiplicity of the unique knots in the knot vector.
  inline torch::Tensor get_multiplicity() const noexcept
  {
    return std::get<1>(get_unique_knots_and_multiplicity());
  }
  /// @return The number of basis functions in this basis.
  inline int64_t get_size() const noexcept
  {
    return knotvector_.size(0) - degree_ - 1;
  }
  /// @return The device used by this object.
  inline torch::Device get_device() const noexcept
  {
    return knotvector_.device();
  }
  /// @return The data type of the knot vector.
  inline torch::ScalarType get_dtype() const noexcept
  {
    return knotvector_.scalar_type();
  }
};

std::ostream& operator<<(std::ostream& os, const BSplineBasis& p);

} // namespace ttnte::cad
