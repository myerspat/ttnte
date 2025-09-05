#include "ttnte/linalg/csr_operator.hpp"
#include <iostream>
#include <stdexcept>
#include <torch/extension.h>

namespace ttnte::linalg {

// =================================================
// Public constructors
// =================================================
CSROperator::CSROperator(const torch::Tensor& tensor)
{
  // Check that the tensor is 2-D
  if (tensor.ndimension() > 2) {
    throw std::runtime_error("Tensors must be 2-D for CSROperator");
  }

  // Convert tensor to CSR if not already
  tensor_ =
    (tensor.layout() != torch::kSparseCsr) ? tensor.to_sparse_csr() : tensor;
}

// =================================================
// Public methods
// =================================================
torch::Tensor CSROperator::matvec(const torch::Tensor& x) const
{
  return torch::matmul(tensor_, x.flatten());
}

void CSROperator::cuda(const int64_t idx)
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

void CSROperator::cpu()
{
  // Get device
  torch::Device device(torch::kCPU);

  // Send tensor to CPU
  tensor_ = tensor_.to(device);
}

} // namespace ttnte::linalg
