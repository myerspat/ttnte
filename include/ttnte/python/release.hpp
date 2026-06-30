#pragma once

#include <memory>

namespace ttnte::python {

class Release {
private:
  /// Implementation and pointer to the scoped GIL in pybind11
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

public:
  Release();
  ~Release();
};

} // namespace ttnte::python
