# --- external/CMakeLists.txt ---
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# MPI is non-negotiable for ParMETIS
find_package(MPI REQUIRED)

# ===========================================================================
# GKlib, METIS, ParMETIS
# ===========================================================================

# These variables are EXACTLY what the internal METIS/ParMETIS files look for
set(GKLIB_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/GKlib"
    CACHE PATH "" FORCE)
set(METIS_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/METIS"
    CACHE PATH "" FORCE)
set(METIS_TPL_SOURCES "") # Some versions need this initialized

# Force the definitions globally so ParMETIS definitely gets them
add_definitions(-DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=64)

# Defeat the hardcoded "-Werror" in GKlib/METIS that causes the macro
# redefinition crash
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-error)
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(METIS_LIBRARY_TYPE
    STATIC
    CACHE STRING "" FORCE)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/GKlib
  ${CMAKE_CURRENT_SOURCE_DIR}/GKlib/include
  ${CMAKE_CURRENT_SOURCE_DIR}/GKlib/src
  ${CMAKE_CURRENT_SOURCE_DIR}/METIS/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ParMETIS/include)

# Build exactly what we need: The raw libraries, nothing else.
add_subdirectory(GKlib)
add_subdirectory(METIS/libmetis ${CMAKE_CURRENT_BINARY_DIR}/metis)
add_subdirectory(ParMETIS/libparmetis ${CMAKE_CURRENT_BINARY_DIR}/parmetis)

target_link_libraries(parmetis PRIVATE MPI::MPI_C)

# EXPORT a single interface target to make your life easier in /src
add_library(partitioning_suite INTERFACE)
target_link_libraries(partitioning_suite INTERFACE parmetis metis GKlib
                                                   MPI::MPI_C)
target_include_directories(
  partitioning_suite
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/ParMETIS/include
            ${CMAKE_CURRENT_SOURCE_DIR}/METIS/include
            ${CMAKE_CURRENT_SOURCE_DIR}/GKlib)

# Force the definitions globally so ParMETIS definitely gets them
target_compile_definitions(partitioning_suite INTERFACE IDXTYPEWIDTH=64
                                                        REALTYPEWIDTH=64)

# ===========================================================================
# torchTT
# ===========================================================================
add_library(torchtt_headers INTERFACE)
target_include_directories(
  torchtt_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/torchTT/torchtt/cpp)
target_compile_definitions(torchtt_headers INTERFACE USE_CUDA)
