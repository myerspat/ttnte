#include "ttnte/linalg/tt_operator.hpp"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/qr.h>
#include <ATen/ops/svd.h>
#include <iostream>
#include <ostream>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;
using namespace torch::indexing;

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
  : TTOperator(tt.attr("cores").cast<std::vector<torch::Tensor>>())
{}

TTOperator::TTOperator(const std::vector<torch::Tensor>& cores) : cores_(cores)
{
  // Check cores are actually for an operator
  for (size_t i = 0; i < cores_.size(); i++) {
    if (cores_[i].ndimension() != 4 &&
        ((cores_[i].ndimension() != 3) && (i == 0 || i == cores_.size() - 1))) {
      throw std::runtime_error(
        "Tensors must be 4-D for an operator or with 3-D end tensors");
    }
  }

  // Squeeze the useless "ranks" at the end
  if (cores_[0].ndimension() == 4) {
    cores_[0] = torch::squeeze(cores_[0], 0);
  }
  if (cores_[cores_.size() - 1].ndimension() == 4) {
    cores_[cores_.size() - 1] = torch::squeeze(
      cores_[cores_.size() - 1], cores_[cores_.size() - 1].ndimension() - 1);
  }

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
  return (this->scale() != 1.0)
           ? this->scale() * tensor_bank[0].reshape(x.sizes()).contiguous()
           : tensor_bank[0].reshape(x.sizes()).contiguous();
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

std::shared_ptr<Operator> TTOperator::add_(
  const std::shared_ptr<Operator>& other)
{
  auto throw_error = []() {
    throw std::runtime_error(
      "Both operators must be TTOperators of the same shape");
  };

  if (auto other_tt = std::dynamic_pointer_cast<TTOperator>(other)) {
    // Check number of dimensions
    if (this->num_cores() != other_tt->num_cores()) {
      throw_error();
    }

    // Check shape within each core
    const auto& this_shape = this->shape();
    const auto& other_shape = other_tt->shape();
    for (size_t i = 0; i < this_shape.size(); i++) {
      if (std::get<0>(this_shape[i]) != std::get<0>(other_shape[i]) ||
          std::get<1>(this_shape[i]) != std::get<1>(other_shape[i])) {
        throw_error();
      }
    }

    // Build cores
    const auto& this_ranks = this->ranks();
    const auto& other_ranks = other_tt->ranks();

    // First core
    cores_[0] = torch::cat(
      {(this->scale() != 1.0) ? (this->scale() * this->cores_.front())
                              : this->cores_.front(),
        (other_tt->scale() != 1.0)
          ? (other_tt->scale() * other_tt->cores().front())
          : other_tt->cores().front()},
      2);
    this->set_scale(1.0);

    // Do middle cores
    for (size_t i = 1; i < this->num_cores() - 1; i++) {
      // Ranks of new core
      auto rl = this_ranks[i - 1] + other_ranks[i - 1];
      auto rr = this_ranks[i] + other_ranks[i];

      // Add initial tensor
      torch::Tensor core = torch::zeros(
        {rl, std::get<0>(this_shape[i]), std::get<1>(this_shape[i]), rr});

      // Copy this core in
      core.slice(0, 0, this_ranks[i - 1])
        .slice(3, 0, this_ranks[i])
        .copy_(this->cores_[i]);

      // Copy other core in
      core.slice(0, this_ranks[i - 1], rl)
        .slice(3, this_ranks[i], rr)
        .copy_(other_tt->cores()[i]);

      cores_[i] = core;
    }

    // Do final core
    cores_[num_cores() - 1] =
      torch::cat({this->cores_.back(), other_tt->cores().back()}, 0);

    return ptr();
  }
  throw_error();
  return nullptr;
}

void TTOperator::lr_orthogonalize(std::optional<int64_t> gpu_idx)
{
  // Put on GPU if requested
  if (gpu_idx.has_value()) {
    this->cuda(gpu_idx.value());
  }

  // Get shape of cores without ranks
  const auto& shape = this->shape();
  auto ranks = this->ranks();

  // Add first and last for ranks
  ranks.reserve(ranks.size() + 2);
  ranks.insert(ranks.begin(), 1);
  ranks.insert(ranks.end(), 1);

  for (size_t i = 0; i < num_cores() - 1; i++) {
    // Get left core
    auto& lcore = cores_[i];

    // Reshape
    lcore = lcore.reshape(
      {ranks[i] * std::get<0>(shape[i]) * std::get<1>(shape[i]), -1});

    // Run QR
    const auto& [Q, R] = at::qr(lcore);
    lcore = torch::reshape(
      Q, {ranks[i], std::get<0>(shape[i]), std::get<1>(shape[i]), -1});

    // Get right core
    auto& rcore = cores_[i + 1];

    // Reshape and apply R
    rcore = torch::reshape(torch::matmul(R, rcore.reshape({ranks[i + 1], -1})),
      {Q.size(1), std::get<0>(shape[i + 1]), std::get<1>(shape[i + 1]),
        ranks[i + 2]});

    // Update ranks
    ranks[i + 1] = Q.size(1);

    // Make sure cores are being updated
    cores_[i] = lcore;
    cores_[i + 1] = rcore;
  }

  // Remove rank 1 on ends
  cores_.front().squeeze_(0);
  cores_.back().squeeze_(3);

  // Remove from GPU
  if (gpu_idx.has_value()) {
    this->cpu();
  }
}

std::shared_ptr<TTOperator> TTOperator::round(
  const double& eps, const int64_t& max_rank, std::optional<int64_t> gpu_idx)
{
  // Clone this TT
  if (num_cores() == 1) {
    return this->typed_ptr();
  }

  // Refine eps
  double eps_refined = eps / std::sqrt(num_cores() - 1);

  // Place on GPU
  if (gpu_idx.has_value()) {
    this->cuda(gpu_idx.value());
  }

  // Left to right orthogonalize
  this->lr_orthogonalize();

  // Iterate through cores from right to left
  auto cores = this->cores();
  auto shape = this->shape();
  auto ranks = this->ranks();
  ranks.reserve(ranks.size() + 2);
  ranks.insert(ranks.begin(), 1);
  ranks.insert(ranks.end(), 1);
  for (size_t i = num_cores() - 1; i > 0; i--) {
    // Get left and right cores
    auto& rcore = cores[i];
    auto& lcore = cores[i - 1];

    // Reshape
    rcore = rcore.reshape({ranks[i], -1});
    lcore = lcore.reshape({-1, ranks[i]});

    // Run SVD
    torch::Tensor u, s, v;
    if (rcore.size(0) < 10 * rcore.size(1)) {
      const auto& result = at::svd(rcore, true);
      u = std::get<0>(result);
      s = std::get<1>(result);
      v = std::get<2>(result);
    } else {
      const auto& result = at::svd(rcore.t(), true);
      u = std::get<2>(result).t_();
      s = std::get<1>(result);
      v = std::get<0>(result).t_();
    }

    // Get new rank
    double snorm = s.norm().item<double>();
    double core_eps = snorm * eps_refined;
    if (snorm == 0) {
      ranks[i] = 1;
    } else if (eps_refined == 0) {
      ranks[i] = s.size(0);
    } else {
      // Calculate energy of each singular value and find where it becomes
      // less than core_eps
      const auto& indices =
        torch::nonzero(torch::cumsum(s.abs().flip(0).pow(2), 0).flip(0) <
                       (core_eps * core_eps));

      if (indices.numel() > 0) {
        ranks[i] = indices[0].item<int64_t>();
      } else {
        ranks[i] = s.size(0);
      }
      ranks[i] = std::min(max_rank, (ranks[i] > 0) ? ranks[i] : 1);
    }

    // Chop, linear algebra, reshapes
    rcore = v.index({Slice(0, ranks[i], 1), Slice()})
              .reshape({ranks[i], std::get<0>(shape[i]), std::get<1>(shape[i]),
                ranks[i + 1]});
    lcore =
      at::linalg_multi_dot({lcore, u.index({Slice(), Slice(0, ranks[i], 1)}),
                             torch::diag(s.index({Slice(0, ranks[i], 1)}))})
        .reshape({ranks[i - 1], std::get<0>(shape[i - 1]),
          std::get<1>(shape[i - 1]), ranks[i]});
  }

  // Remove rank 1 on ends
  cores.front().squeeze_(0);
  cores.back().squeeze_(3);

  // Remove from GPU
  if (gpu_idx.has_value()) {
    this->cpu();
  }

  return std::make_shared<TTOperator>(cores);
}

torch::Tensor TTOperator::to_dense() const
{
  torch::Tensor dense = cores_[0];
  std::vector<int64_t> axes;
  axes.reserve(2 * num_cores());
  axes.push_back(0);
  axes.push_back(1);

  for (size_t i = 1; i < num_cores(); i++) {
    dense = torch::tensordot(dense, cores_[i], -1, 0);

    axes.insert(axes.begin() + i, i * 2);
    axes.push_back(i * 2 + 1);
  }

  return torch::permute(dense, axes);
}

// =================================================
// Friend methods
// =================================================
std::ostream& operator<<(std::ostream& os, const TTOperator& op)
{
  os << "TTOPerator";
  return os;
}

} // namespace ttnte::linalg
