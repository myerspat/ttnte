#include "ttnte/python/acquire.hpp"
#include <torch/extension.h>

namespace ttnte::python {

struct Acquire::Impl {
  py::gil_scoped_acquire acquire;
};

Acquire::Acquire() : pimpl_(std::make_unique<Impl>()) {}
Acquire::~Acquire() = default;

} // namespace ttnte::python
