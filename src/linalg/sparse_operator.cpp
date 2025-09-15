#include "ttnte/linalg/sparse_operator.hpp"
#include <iostream>
#include <stdexcept>
#include <torch/extension.h>

namespace ttnte::linalg {

// =================================================
// Public constructors
// =================================================
SparseOperator::SparseOperator(const torch::Tensor& tensor) : Operator()
{
  // Check that the tensor is 2-D
  if (tensor.ndimension() > 2) {
    throw std::runtime_error("Tensors must be 2-D for SparseOperator");
  }

  // Check format
  if (!tensor.is_sparse()) {
    tensor_ = tensor.to_sparse_csr();
  } else if (tensor.layout() == torch::kSparse ||
             tensor.layout() == torch::kSparseCsr) {
    tensor_ = tensor;
  } else {
    throw std::runtime_error("The tensor must be dense, COO, or CSR format");
  }
}

// =================================================
// Public methods
// =================================================
torch::Tensor SparseOperator::apply(const torch::Tensor& x) const
{
  return (this->scale() != 1.0)
           ? this->scale() * torch::matmul(tensor_, x.reshape({-1, 1}))
                               .reshape(x.sizes())
                               .contiguous()
           : torch::matmul(tensor_, x.reshape({-1, 1}))
               .reshape(x.sizes())
               .contiguous();
}

void SparseOperator::cuda(const int64_t idx)
{
  if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
    // Get device
    torch::Device device(torch::kCUDA, idx);

    // Send tensor to GPU
    tensor_ = tensor_.to(device);
  } else {
    std::cout << "WARNING: CUDA not available" << std::endl;
  }
}

void SparseOperator::cpu()
{
  // Get device
  torch::Device device(torch::kCPU);

  // Send tensor to CPU
  tensor_ = tensor_.to(device);
}

std::shared_ptr<Operator> SparseOperator::add_(
  const std::shared_ptr<Operator>& other)
{
  if (auto sparse_other = std::dynamic_pointer_cast<SparseOperator>(other)) {
    // Combine tensors
    this->tensor_ = ((this->scale() != 1.0) ? (this->scale() * this->tensor_)
                                            : this->tensor_) +
                    ((sparse_other->scale() != 1.0)
                        ? (sparse_other->scale() * sparse_other->tensor())
                        : sparse_other->tensor());
    this->set_scale(1.0);
    return ptr();
  }
  throw std::runtime_error("Both operators must be a SparseOperator to add");
}

torch::Tensor SparseOperator::to_dense() const
{
  return tensor_.to_dense();
}

} // namespace ttnte::linalg
