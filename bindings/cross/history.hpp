#pragma once

#include "ttnte/cross/layout_cross.hpp"
#include <pybind11/pybind11.h>
#include <vector>

namespace ttnte::bindings {

/// @brief A cross run's half-sweeps as a list of dicts (LayoutCross.history
/// and CrossRun.history).
pybind11::list sweep_history(
  const std::vector<cross::LayoutCross::SweepRecord>& history);

} // namespace ttnte::bindings
