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

# Profiling flags
if(TTNTE_ENABLE_PROFILE)
  list(APPEND flags -g -fno-omit-frame-pointer)
endif()

# Optimization flags
if(TTNTE_OPTIMIZED)
  list(APPEND flags -O3 -march=native)
endif()

# Show flags
message(STATUS "Compiling ttnte C++ backend")
message(STATUS "Project source directory: ${PROJECT_SOURCE_DIR}")
message(STATUS "ttnte C++ flags: ${flags}")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")

# ================================================================================
# Get dependencies

list(APPEND CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX}/shape/cmake/Torch)

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH
             "${TORCH_INSTALL_PREFIX}/lib")

message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch Include: ${TORCH_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ================================================================================
# Include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# ================================================================================
# Create ttnte.linalg module
add_library(linalg MODULE ${PROJECT_SOURCE_DIR}/bindings/linalg.cpp)
target_include_directories(
  linalg
  PRIVATE ${TORCH_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/bindings
          ${Python3_INCLUDE_DIR})

# Link libraries
target_link_libraries(linalg PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Set output name
set_target_properties(
  linalg
  PROPERTIES OUTPUT_NAME "linalg"
             PREFIX "cpp/"
             SUFFIX ".so")
