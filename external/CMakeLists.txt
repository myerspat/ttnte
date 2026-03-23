# --- external/CMakeLists.txt ---
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# MPI is non-negotiable for ParMETIS
find_package(MPI REQUIRED)

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

# Ensure ParMETIS finds the METIS headers BEFORE it compiles
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/METIS/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/GKlib)
link_libraries(MPI::MPI_C)

# Add them in order. ParMETIS expects METIS to be configured first.
add_subdirectory(GKlib)
add_subdirectory(METIS)
add_subdirectory(ParMETIS)

# EXPORT a single interface target to make your life easier in /src
add_library(partitioning_suite INTERFACE)
target_link_libraries(partitioning_suite INTERFACE parmetis metis GKlib
                                                   MPI::MPI_C)
target_include_directories(
  partitioning_suite
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/ParMETIS/include
            ${CMAKE_CURRENT_SOURCE_DIR}/METIS/include
            ${CMAKE_CURRENT_SOURCE_DIR}/GKlib)
