#pragma once

#include <torch/extension.h>

namespace ttnte::utils {

// Unpack tensor from MPI communication
template<typename data_t, typename shape_t>
torch::Tensor unpack_tensor(
  data_t* data_ptr, const shape_t& shape, const torch::TensorOptions& options)
{
  int64_t size = 1;

  for (const auto& dim_size : shape) {
    size *= dim_size;
  }

  return torch::from_blob(data_ptr, {size}, options).clone().reshape(shape);
}

} // namespace ttnte::utils
