#pragma once

#include <memory>
#include <torch/extension.h>

namespace ttnte::parallel {

enum class BoundaryType : int8_t { TENSOR_TRAIN = 0 };

class BoundaryBuffer : public std::enable_shared_from_this<BoundaryBuffer> {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<BoundaryBuffer>;

protected:
  // =================================================================
  // Protected data
  /// The state vector for the boundary.
  torch::Tensor buffer_;

public:
  // =================================================================
  // Public types
};

} // namespace ttnte::parallel
