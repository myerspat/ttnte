#pragma once

#include "ttnte/utils/io_formatting.hpp"
#include <sstream>
#include <stdexcept>
#include <string>

namespace ttnte::utils {

template<typename T>
inline std::runtime_error runtime_error(
  const T& caller, const std::string& method, const std::string& message)
{
  std::stringstream ss;
  ss << "Error in " << method << " with label " << caller.get_label() << ":\n"
     << indent_message(message, 2);

  return std::runtime_error(ss.str());
}

static inline std::runtime_error runtime_error(
  const std::string& method, const std::string& message)
{
  std::stringstream ss;
  ss << "Error in " << method << ":\n" << indent_message(message, 2);

  return std::runtime_error(ss.str());
}

} // namespace ttnte::utils
