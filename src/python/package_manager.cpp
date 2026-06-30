#include "ttnte/python/package_manager.hpp"

namespace ttnte::python {

PackageManager& PackageManager::instance()
{
  static PackageManager instance;
  return instance;
}

PackageManager::PackageManager()
{
  numpy = py::module_::import("numpy");
  torchtt = py::module_::import("torchtt");
}

void PackageManager::clear()
{
  // Clear modules
  numpy = py::module_();
  torchtt = py::module_();
}

} // namespace ttnte::python
