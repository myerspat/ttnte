#pragma once

#include <torch/extension.h>

namespace ttnte::python {

/// @brief Singleton that owns the Python modules. This class is meant to be
/// hidden from the rest of the codebase so this is only to be included in .cpp
/// files.
class PackageManager {
public:
  // Modules
  py::module_ torchtt;

  /// @return Get the instance of the singleton manager.
  static PackageManager& instance();

  /// @brief Delete all references to Python modules. This method is bound and
  /// called by atexit.
  void clear();

private:
  PackageManager();
};

} // namespace ttnte::python
