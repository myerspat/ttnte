# Require 3.14 or higher
cmake_minimum_required(VERSION 3.14)

# Project name and version
project(
  ttnte
  VERSION 0.0.0
  LANGUAGES CXX)

# C++ version minimum
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

# ================================================================================
# Compiler options

# Supported options
option(TTNTE_PROFILE "Compile with profiling flags" OFF)
option(TTNTE_OPTIMIZED "Compile with optimization" OFF)
option(USE_CUDA "Use CUDA configuration of torch" ON)

# Profiling flags
if(TTNTE_ENABLE_PROFILE)
  list(APPEND flags -g -fno-omit-frame-pointer)
endif()

# Optimization flags
if(TTNTE_OPTIMIZED)
  list(APPEND flags -O3 -march=native)
endif()

# Add to compile definitions
list(APPEND flags _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})

# Show flags
message(STATUS "Compiling ttnte C++ backend")
message(STATUS "Project source directory: ${PROJECT_SOURCE_DIR}")
message(STATUS "ttnte C++ flags: ${flags}")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")

# Set compile flags as global for now add_compile_definitions(${flags})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flags}")

# ================================================================================
# Get dependencies
list(APPEND CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX}/share/cmake)

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_package(Torch REQUIRED)

if(NOT USE_CUDA)
  # Disable CUDA
  set(CMAKE_DISABLE_FIND_PACKAGE_CUDA TRUE)
  set(TORCH_USE_CUDA
      OFF
      CACHE BOOL "Torch: disable CUDA" FORCE)
  set(CAFFE2_USE_CUDA
      OFF
      CACHE BOOL "Caffe2: disable CUDA" FORCE)
  set(Torch_CUDA_FOUND
      OFF
      CACHE BOOL "Torch CUDA found" FORCE)
  message(STATUS "Compile pytorch without CUDA")
endif()

find_library(TORCH_PYTHON_LIBRARY torch_python PATH
             "${TORCH_INSTALL_PREFIX}/lib")

message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch Include: ${TORCH_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ================================================================================
# Create modules
add_subdirectory(bindings/linalg)
