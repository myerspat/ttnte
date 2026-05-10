cmake_minimum_required(VERSION 3.26)

# Project setup
project(
  ttnte
  VERSION 0.0.0
  LANGUAGES C CXX)

# Standard C++20/CUDA20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Conditionally enable CUDA
option(USE_CUDA "Use CUDA configuration" ON)
if(USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
  set(CMAKE_CUDA_ARCHITECTURES
      ""
      CACHE STRING "" FORCE)
else()
  set(TORCH_USE_CUDA OFF)
  message(STATUS "Compiling without CUDA")
endif()

# Suppress Kineto warning
set(KINETO_LIBRARY "")

# Prevent RPATH confusion
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

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

# Find where PyTorch hides its LAPACK/MKL
find_library(
  TORCH_LAPACK_LIB
  NAMES torch_cpu mkl_rt caffe2_lapack
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH)

message(STATUS "Found Torch LAPACK provider: ${TORCH_LAPACK_LIB}")

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
# Other external libraries in external
# ==============================================================================
add_subdirectory(external)
add_compile_options(-Wno-narrowing)

# ==============================================================================
# FLAGS AND OPTIONS
# ==============================================================================
option(TTNTE_OPTIMIZED "Compile with optimization" OFF)

if(TTNTE_OPTIMIZED)
  add_compile_options(-O3 -march=native)
else()
  add_compile_options(-g -fno-omit-frame-pointer)
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
# SUBDIRECTORIES
# ==============================================================================
add_subdirectory(bindings/utils)
add_subdirectory(bindings/parallel)
add_subdirectory(bindings/xs)
add_subdirectory(bindings/mesh)
add_subdirectory(bindings/cad)
add_subdirectory(bindings/physics)
add_subdirectory(bindings/driver)
add_subdirectory(bindings/linalg)
add_subdirectory(bindings/task)
add_subdirectory(bindings/solvers)
add_subdirectory(bindings/python)

# ==============================================================================
# C++ side
# ==============================================================================
# Create single static binary and link to dependencies
add_library(
  ttnte_static OBJECT
  ${UTILS_SRC}
  ${PARALLEL_SRC}
  ${XS_SRC}
  ${CAD_SRC}
  ${TASK_SRC}
  ${LINALG_SRC}
  ${SOLVERS_SRC}
  ${PY_SRC})
set_target_properties(ttnte_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ttnte_static PUBLIC partitioning_suite ${TORCH_LIBRARIES}
                                          MPI::MPI_CXX torchtt_headers)
target_include_directories(
  ttnte_static
  PUBLIC ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
         ${PROJECT_SOURCE_DIR}/src ${Python3_INCLUDE_DIRS}
         ${MPI_CXX_INCLUDE_DIRS} partitioning_suite)

find_package(LAPACK)
if(LAPACK_FOUND)
  target_link_libraries(ttnte_static PUBLIC ${LAPACK_LIBRARIES})
else()
  # If not found, we attempt to use the one Conda usually ships
  target_link_libraries(ttnte_static PUBLIC "-llapack -lblas")
endif()

# ==============================================================================
# Python side
# ==============================================================================
# Create python bindings library
add_library(
  ttnte_python MODULE
  ${PROJECT_SOURCE_DIR}/bindings/bindings.cpp
  ${XS_BINDINGS}
  ${CAD_BINDINGS}
  ${MESH_BINDINGS}
  ${PHYSICS_BINDINGS}
  ${PARALLEL_BINDINGS}
  ${DRIVER_BINDINGS}
  ${LINALG_BINDINGS}
  ${TASK_BINDINGS}
  ${SOLVERS_BINDINGS}
  ${PY_BINDINGS})

# Link python bindings to the static
target_link_libraries(ttnte_python PRIVATE ttnte_static ${TORCH_PYTHON_LIBRARY})

target_include_directories(
  ttnte_python
  PRIVATE ${TORCH_INCLUDE_DIRS}
          ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src
          ${Python3_INCLUDE_DIRS}
          ${XS_BINDINGS_DIR}
          ${CAD_BINDINGS_DIR}
          ${MESH_BINDINGS_DIR}
          ${PHYSICS_BINDINGS_DIR}
          ${PARALLEL_BINDINGS_DIR}
          ${DRIVER_BINDINGS_DIR}
          ${LINALG_BINDINGS_DIR}
          ${TASK_BINDINGS_DIR}
          ${SOLVERS_BINDINGS_DIR}
          ${PY_BINDINGS_DIR})

if(APPLE)
  set_target_properties(ttnte_python PROPERTIES LINK_FLAGS
                                                "-undefined dynamic_lookup")
endif()

set_target_properties(
  ttnte_python
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/ttnte/cpp"
             OUTPUT_NAME "ttnte_python"
             PREFIX "")

set_target_properties(
  ttnte_python PROPERTIES INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
                          BUILD_WITH_INSTALL_RPATH TRUE)

# ==============================================================================
# Install
# ==============================================================================
install(TARGETS ttnte_python DESTINATION ttnte/cpp)
