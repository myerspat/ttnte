cmake_minimum_required(VERSION 3.26)

# Project setup
project(
  ttnte
  VERSION 0.0.0
  LANGUAGES CXX CUDA)

# Standard C++20/CUDA20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major) # or 'native' if you're only on one
                                          # machine
endif()

# ==============================================================================
# Requirements
# ==============================================================================
find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)
find_package(MPI REQUIRED)

# ==============================================================================
# CONFIGURE TORCH PATHS We ask Python for the exact CMake path. This prevents
# linking to the wrong system-level libtorch and fixes the "cycle" warning.
# ==============================================================================
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PATH
  RESULT_VARIABLE TORCH_PROBE_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT TORCH_PROBE_RESULT EQUAL 0 AND DEFINED ENV{CONDA_PREFIX})
  message(STATUS "Python probe failed, falling back to CONDA_PREFIX")
  set(TORCH_CMAKE_PATH
      "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/torch/share/cmake")
endif()

if(TORCH_CMAKE_PATH)
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
  message(STATUS "Torch CMake Prefix: ${TORCH_CMAKE_PATH}")
else()
  message(
    FATAL_ERROR "Could not locate Torch. Please set -DTorch_DIR manually.")
endif()

# Find Torch (Now it will definitely find the Python version)
find_package(Torch REQUIRED)

# Manually find the Python-specific Torch library (libtorch_python.so) We look
# in the parent directory of the cmake path (usually site-packages/torch/lib)
get_filename_component(TORCH_INSTALL_PREFIX "${TORCH_CMAKE_PATH}/../.."
                       ABSOLUTE)
find_library(
  TORCH_PYTHON_LIBRARY torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH)

# ==============================================================================
# CONFIGURE CUTENSOR
# ==============================================================================
# # Define where cuTensor is likely located (adjust if installed elsewhere)
# if(DEFINED ENV{CONDA_PREFIX}) set(CUTENSOR_SEARCH_PATHS "$ENV{CONDA_PREFIX}"
# "/usr/local/cuda-12.9") else() set(CUTENSOR_SEARCH_PATHS
# "/usr/local/cuda-12.9") endif()
#
# find_path( CUTENSOR_INCLUDE_DIR cutensor.h PATHS ${CUTENSOR_SEARCH_PATHS}
# PATH_SUFFIXES include)
#
# find_library( CUTENSOR_LIBRARY cutensor PATHS ${CUTENSOR_SEARCH_PATHS}
# PATH_SUFFIXES lib64 lib)
#
# if(CUTENSOR_INCLUDE_DIR AND CUTENSOR_LIBRARY) message(STATUS "Found cuTensor:
# ${CUTENSOR_LIBRARY}") if(NOT TARGET nvidia::cutensor)
# add_library(nvidia::cutensor SHARED IMPORTED) set_target_properties(
# nvidia::cutensor PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
# "${CUTENSOR_INCLUDE_DIR}" IMPORTED_LOCATION "${CUTENSOR_LIBRARY}") endif()
# else() message( FATAL_ERROR "cuTensor not found. Searched in:
# ${CUTENSOR_SEARCH_PATHS}") endif()

# ==============================================================================
# CHECK ABI COMPATIBILITY Ensure we use the same std::string ABI as the compiled
# Torch
# ==============================================================================
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE _GLIBCXX_USE_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})

# ==============================================================================
# FLAGS AND OPTIONS
# ==============================================================================
option(TTNTE_OPTIMIZED "Compile with optimization" OFF)
option(USE_CUDA "Use CUDA configuration" ON)

if(TTNTE_OPTIMIZED)
  add_compile_options(-O3 -march=native)
else()
  add_compile_options(-g -fno-omit-frame-pointer)
endif()

if(NOT USE_CUDA)
  set(TORCH_USE_CUDA OFF)
  message(STATUS "Compiling without CUDA")
endif()

# ==============================================================================
# 1. LOGGING
# ==============================================================================
message(STATUS "------------------------------------------------")
message(STATUS "  ttnte Build Configuration")
message(STATUS "------------------------------------------------")
message(STATUS "  Python: ${Python3_EXECUTABLE}")
message(STATUS "  Torch Prefix: ${TORCH_INSTALL_PREFIX}")
message(STATUS "  Torch Libs: ${TORCH_LIBRARIES}")
message(STATUS "  Torch Python Lib: ${TORCH_PYTHON_LIBRARY}")
message(STATUS "  MPI Found: ${MPI_CXX_FOUND}")
message(STATUS "  ABI Version: ${_GLIBCXX_USE_CXX11_ABI}")
# message(STATUS "  cuTensor Lib: ${CUTENSOR_LIBRARY}")
message(STATUS "------------------------------------------------")

# ==============================================================================
# 1. SUBDIRECTORIES
# ==============================================================================
add_subdirectory(bindings/utils)
add_subdirectory(bindings/cad)

install(TARGETS utils_python cad_python DESTINATION ttnte/cpp)
