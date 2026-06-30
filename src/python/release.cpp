#include "ttnte/python/release.hpp"
#include <torch/extension.h>

namespace ttnte::python {

struct Release::Impl {
  py::gil_scoped_release release;
};

Release::Release() : pimpl_(std::make_unique<Impl>()) {}
Release::~Release() = default;

} // namespace ttnte::python
