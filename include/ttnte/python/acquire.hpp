#pragma once

#include <memory>
namespace ttnte::python {

/// @brief Base class for acquiring the GIL and accessing Python packages in the
/// singleton PackageManager.
class Acquire {
protected:
  /// Implementation and pointer to the scoped GIL in pybind11
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

public:
  Acquire();
  virtual ~Acquire();
};

} // namespace ttnte::python
