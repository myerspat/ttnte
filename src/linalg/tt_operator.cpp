#include "ttnte/linalg/tt_operator.hpp"
#include <ATen/core/TensorBody.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

namespace ttnte::linalg {

// =================================================
// Public constructors
// =================================================
TTOperator::TTOperator(const std::vector<torch::Tensor>& cores,
  const std::vector<ContractionStep>& csteps, const std::vector<int64_t>& lidxs,
  const std::vector<int64_t>& ridxs)
  : Operator(), cores_(cores), csteps_(csteps), lidxs_(lidxs), ridxs_(ridxs)
{
  // Check lengths
  assert(csteps.size() == lidxs.size());
  assert(csteps.size() == ridxs.size());

  // Check cores are actually for an operator
  for (int i = 0; i < cores.size(); i++) {
    if ((cores[i].ndimension() + ((i == 0 || i == cores.size() - 1) ? 1 : 0)) <
        4) {
      throw std::runtime_error(
        "The first and last tensor must be 3-D and the rest 4-D");
    }
  }

  // Iterate through each core to make sure its contiguous
  for (auto& core : cores_) {
    if (!core.is_contiguous()) {
      core = core.contiguous();
    }
  }
}

TTOperator::TTOperator(const py::object& tt)
{
  // Get vector of cores
  cores_ = tt.attr("cores").cast<std::vector<torch::Tensor>>();

  // Check cores are actually for an operator
  for (const auto& core : cores_) {
    if (core.ndimension() != 4) {
      throw std::runtime_error("Tensors must be 4-D for an operator");
    }
  }

  // Squeeze the useless "ranks" at the end
  cores_[0] = torch::squeeze(cores_[0], 0);
  cores_[cores_.size() - 1] = torch::squeeze(
    cores_[cores_.size() - 1], cores_[cores_.size() - 1].ndimension() - 1);

  // Iterate through each core to make sure its contiguous
  for (auto& core : cores_) {
    if (!core.is_contiguous()) {
      core = core.contiguous();
    }
  }

  // Get all shapes
  std::vector<at::IntArrayRef> shape;
  shape.reserve(cores_.size());
  for (const auto& core : cores_) {
    shape.push_back(core.sizes());
  }

  // Get contraction steps, left indices, and right indices
  const auto& [csteps, lidxs, ridxs] =
    py::module::import("ttnte.cpp.utils._torchtt2cpp")
      .attr("torchtt2cpp")(shape)
      .cast<std::tuple<std::vector<ContractionStep>, std::vector<int64_t>,
        std::vector<int64_t>>>();

  csteps_ = csteps;
  lidxs_ = lidxs;
  ridxs_ = ridxs;
}

// =================================================
// Private methods
// =================================================
std::vector<int64_t> TTOperator::get_shape(
  const std::size_t& i, const std::size_t& j) const
{
  // Initialize shape
  std::vector<int64_t> shape;
  shape.reserve(cores_.size());

  // Add shapes
  shape.push_back(cores_[0].size(i));
  for (size_t k = 1; k < cores_.size(); k++) {
    shape.push_back(cores_[k].size(j));
  }

  return shape;
}

// =================================================
// Public methods
// =================================================
torch::Tensor TTOperator::apply(const torch::Tensor& x) const
{
  const auto& shape = input_shape();
  assert(torch::numel(x) == std::accumulate(shape.cbegin(), shape.cend(), 1,
                              std::multiplies<int64_t> {}));

  // Create tensor bank with all tensors
  std::vector<torch::Tensor> tensor_bank = cores_;
  tensor_bank.push_back(x.reshape(input_shape()));

  // Loop through contraction steps
  for (size_t i = 0; i < csteps_.size(); ++i) {
    // Get tensors
    const auto& ltensor = tensor_bank[lidxs_[i]];
    const auto& rtensor = tensor_bank[ridxs_[i]];

    // Check that the tensors are on the same device
    assert(ltensor.device() == rtensor.device());

    // Compute contractions and place in left tensor position
    tensor_bank[std::min(lidxs_[i], ridxs_[i])] =
      csteps_[i].contract(ltensor, rtensor);

    // Erase outer tensor
    tensor_bank.erase(tensor_bank.begin() + std::max(lidxs_[i], ridxs_[i]));
  }

  // Check there is one tensor left in tensor bank
  assert(tensor_bank.size() == 1);
  return tensor_bank[0].reshape(x.sizes());
}

void TTOperator::cuda(const int64_t idx)
{
  if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
    // Get device
    torch::Device device(torch::kCUDA, idx);

    // Send all tensors to GPU
    for (auto& core : cores_) {
      core = core.to(device);
    }
  } else {
    std::cout << "WARNING: CUDA not available" << std::endl;
  }
}

void TTOperator::cpu()
{
  // Get CPU
  torch::Device device(torch::kCPU);

  // Send all tensors to CPU
  for (auto& core : cores_) {
    core = core.to(device);
  }
}

} // namespace ttnte::linalg
