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
  return torch::matmul(tensor_, x.reshape({-1, 1})).reshape(x.sizes());
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

} // namespace ttnte::linalg
