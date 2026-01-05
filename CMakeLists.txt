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

if(NOT TORCH_INSTALL_PREFIX)
  execute_process(
    COMMAND
      "${Python3_EXECUTABLE}" -c
      "import torch, os; print(os.path.abspath(os.path.dirname(torch.__file__)))"
    RESULT_VARIABLE _torch_query_result
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT _torch_query_result EQUAL 0)
    message(FATAL_ERROR "Failed to locate torch via Python")
  endif()
endif()

if(NOT _GLIBCXX_USE_CXX11_ABI)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    RESULT_VARIABLE _torch_query_result
    OUTPUT_VARIABLE _GLIBCXX_USE_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT _torch_query_result EQUAL 0)
    message(FATAL_ERROR "Failed to locate torch via Python")
  endif()
endif()

# Show flags
message(STATUS "Compiling ttnte C++ backend")
message(STATUS "Project source directory: ${PROJECT_SOURCE_DIR}")
message(STATUS "ttnte C++ flags: ${flags}")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")

# Set compile flags as global for now add_compile_definitions(${flags})
string(JOIN " " flags_str ${flags})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flags_str}")

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
