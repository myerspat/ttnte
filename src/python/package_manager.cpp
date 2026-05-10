#include "ttnte/python/package_manager.hpp"

namespace ttnte::python {

PackageManager& PackageManager::instance()
{
  static PackageManager instance;
  return instance;
}

PackageManager::PackageManager()
{
  torchtt = py::module_::import("torchtt");
}

void PackageManager::clear()
{
  // Clear modules
  torchtt = py::module_();
}

} // namespace ttnte::python
