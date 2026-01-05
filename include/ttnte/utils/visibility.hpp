#pragma once

#if defined(_WIN32)
  #define TTNTE_API __declspec(dllexport)
#else
  #define TTNTE_API __attribute__((visibility("default")))
#endif
