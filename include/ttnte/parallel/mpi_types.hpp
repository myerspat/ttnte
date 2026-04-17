#pragma once

#include <cstddef>
#include <cstdint>

namespace ttnte::parallel {

/// @brief Mapping of C++ types to type-erased MPI handles (integers).
template<typename T>
struct std2mpi;

template<>
struct std2mpi<int32_t> {
  static int value();
};

template<>
struct std2mpi<int64_t> {
  static int value();
};

template<>
struct std2mpi<float> {
  static int value();
};

template<>
struct std2mpi<double> {
  static int value();
};

template<>
struct std2mpi<std::byte> {
  static int value();
};

template<typename T>
inline int get_mpi_type()
{
  return std2mpi<T>::value();
}

} // namespace ttnte::parallel
