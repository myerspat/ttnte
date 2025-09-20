#include "ttnte/linalg/fission_operator.hpp"
#include <c10/core/Layout.h>
#include <stdexcept>

namespace ttnte::linalg {

// =================================================
// Public constructors
// =================================================
FissionOperator::FissionOperator(
  const torch::Tensor& F, const torch::Tensor& w_mu, const torch::Tensor& w_eta)
  : Operator(), F_(F), w_mu_(w_mu), w_eta_(w_eta)
{
  assert(F_.ndimension() == 2);
  assert(w_mu_.ndimension() == 1);
  assert(w_eta_.ndimension() == 1);

  const auto& dtype = F_.dtype();
  const auto& device = F_.device();

  if (w_mu_.dtype() != dtype || w_mu_.device() != device ||
      w_eta_.dtype() != dtype || w_eta_.device() != device) {
    throw std::runtime_error(
      "F, w_mu, and w_eta must be the same type and on the same device");
  }

  // Check layout
  if (F_.is_sparse() && F_.layout() != torch::kSparse &&
      F_.layout() != torch::kSparseCsr) {
    throw std::runtime_error("F must be a dense, COO, or CSR 2-D tensor");
  }
}

// =================================================
// Public methods
// =================================================
torch::Tensor FissionOperator::apply(const torch::Tensor& x) const
{
  // Get angular integration shape
  std::vector<int64_t> shape = {4, w_mu_.size(0), w_eta_.size(0), -1};

  // Apply angular integration
  torch::Tensor result =
    torch::einsum("abcd,b,c->d", {x.reshape(shape), w_mu_, w_eta_});

  // Apply fission operator
  return (this->scale() != 1.0)
           ? this->scale() * torch::einsum("abc,d->abcd",
                               {torch::ones({4, w_mu_.size(0), w_eta_.size(0)},
                                  torch::TensorOptions().device(F_.device())),
                                 torch::matmul(F_, result)})
                               .reshape(x.sizes())
                               .contiguous()
           : torch::einsum(
               "abc,d->abcd", {torch::ones({4, w_mu_.size(0), w_eta_.size(0)},
                                 torch::TensorOptions().device(F_.device())),
                                torch::matmul(F_, result)})
               .reshape(x.sizes())
               .contiguous();
}

void FissionOperator::cuda(const int64_t idx)
{
  if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
    // Get device
    torch::Device device(torch::kCUDA, idx);

    // Send to GPU
    F_ = F_.to(device);
    w_mu_ = w_mu_.to(device);
    w_eta_ = w_eta_.to(device);

  } else {
    std::cout << "WARNING: CUDA not available" << std::endl;
  }
}

void FissionOperator::cpu()
{
  // Get device
  torch::Device device(torch::kCPU);

  // Send to CPU
  F_ = F_.to(device);
  w_mu_ = w_mu_.to(device);
  w_eta_ = w_eta_.to(device);
}

std::shared_ptr<Operator> FissionOperator::add_(
  const std::shared_ptr<Operator>& other)
{
  throw std::runtime_error("Combining two FissionOperators is not allowed");
}

std::shared_ptr<Operator> FissionOperator::type(
  const caffe2::TypeMeta& dtype) const
{
  return std::make_shared<FissionOperator>(
    F_.to(dtype), w_mu_.to(dtype), w_eta_.to(dtype));
}

} // namespace ttnte::linalg
