#include "ttnte/linalg/scatter_operator.hpp"
#include <c10/core/Layout.h>
#include <stdexcept>

namespace ttnte::linalg {

// =================================================
// Public constructors
// =================================================
ScatterOperator::ScatterOperator(const std::vector<torch::Tensor>& S,
  const torch::Tensor& Y, const torch::Tensor& w_mu, const torch::Tensor& w_eta)
  : Operator(), S_(S), Y_(Y), w_mu_(w_mu), w_eta_(w_eta)
{
  assert(Y_.ndimension() == 4);
  assert(w_mu_.ndimension() == 1);
  assert(w_eta_.ndimension() == 1);

  for (const auto& s : S_) {
    assert(s.ndimension() == 2);

    // Check layout
    if (s.is_sparse() && s.layout() != torch::kSparse &&
        s.layout() != torch::kSparseCsr) {
      throw std::runtime_error(
        "S must be a vector of dense, COO, or CSR 2-D tensors");
    }
  }
}

// =================================================
// Public methods
// =================================================
torch::Tensor ScatterOperator::apply(const torch::Tensor& x) const
{
  // Get angular integration shape
  std::vector<int64_t> shape = {Y_.size(1), Y_.size(2), Y_.size(3), -1};

  // Apply spherical harmonics and angular integration
  torch::Tensor result =
    torch::einsum("abcd,labc,b,c->ld", {x.reshape(shape), Y_, w_mu_, w_eta_});

  // Calculate zeroth moment
  result.index_put_({0, torch::indexing::Slice()},
    S_[0].matmul(result.index({0, torch::indexing::Slice()})));

  // Iterate through the remaining moments
  int i = 1;
  for (int n = 1; n < S_.size(); n++) {
    for (int m = 0; m < n + 1; m++) {
      result.index_put_({i, torch::indexing::Slice()},
        S_[n].matmul(result.index({i, torch::indexing::Slice()})));
      i++;
    }
  }

  // Compute the outer product with spherical harmonics
  return (this->scale() != 1.0)
           ? this->scale() * torch::einsum("ld,labc->abcd", {result, Y_})
                               .reshape(x.sizes())
                               .contiguous()
           : torch::einsum("ld,labc->abcd", {result, Y_})
               .reshape(x.sizes())
               .contiguous();
}

void ScatterOperator::cuda(const int64_t idx)
{
  if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
    // Get device
    torch::Device device(torch::kCUDA, idx);

    // Send to GPU
    Y_ = Y_.to(device);
    w_mu_ = w_mu_.to(device);
    w_eta_ = w_eta_.to(device);

    for (auto& s : S_) {
      s = s.to(device);
    }
  } else {
    std::cout << "WARNING: CUDA not available" << std::endl;
  }
}

void ScatterOperator::cpu()
{
  // Get device
  torch::Device device(torch::kCPU);

  // Send to CPU
  Y_ = Y_.to(device);
  w_mu_ = w_mu_.to(device);
  w_eta_ = w_eta_.to(device);

  for (auto& s : S_) {
    s = s.to(device);
  }
}

std::shared_ptr<Operator> ScatterOperator::add_(
  const std::shared_ptr<Operator>& other)
{
  throw std::runtime_error("Combining two ScatterOperators is not allowed");
}

} // namespace ttnte::linalg
