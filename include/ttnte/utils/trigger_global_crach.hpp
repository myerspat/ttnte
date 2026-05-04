#pragma once

#include <string>

namespace ttnte::utils {

/// @brief Trigger a crash across all MPI ranks.
/// @param error_msg The error message to be printed to the user.
void trigger_global_crash(const std::string& error_msg);

} // namespace ttnte::utils
