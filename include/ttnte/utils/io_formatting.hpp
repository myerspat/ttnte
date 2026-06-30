#pragma once

#include <string>

namespace ttnte::utils {

static inline std::string indent_message(
  const std::string& msg, size_t spaces = 2)
{
  std::string indentation(spaces, ' ');
  // Prepend indentation to the very first line
  std::string result = indentation;

  for (size_t i = 0; i < msg.length(); i++) {
    result += msg[i];
    // If we hit a newline and it's not the very last character, add spaces
    if (msg[i] == '\n' && i + 1 < msg.length()) {
      result += indentation;
    }
  }
  return result;
}

} // namespace ttnte::utils
