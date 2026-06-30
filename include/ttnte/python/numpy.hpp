#pragma once

#include "ttnte/python/acquire.hpp"
#include <torch/extension.h>

namespace ttnte::python::numpy {

class Acquire : public python::Acquire {
public:
  Acquire();
  ~Acquire() override;

  /// @brief Compute a degree `deg` Gauss-Legendre quadrature.
  std::pair<torch::Tensor, torch::Tensor> leggauss(int64_t deg) const;
};

} // namespace ttnte::python::numpy
