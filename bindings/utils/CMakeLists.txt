# Relevant files from bindings/utils
set(UTILS_BINDINGS bindings.cpp parallel_context.cpp)

# # Relevant files from src set(UTILS_SRC
# ${PROJECT_SOURCE_DIR}/src/utils/bspline_basis.cpp)
#
# # Internal C++ Logic (Logic) add_library(utils_static STATIC ${UTILS_SRC})
#
# # We need PIC even for the static lib because it gets swallowed by a shared #
# module (.so) set_target_properties(utils_static PROPERTIES
# POSITION_INDEPENDENT_CODE ON)
#
# target_include_directories( utils_static PUBLIC ${TORCH_INCLUDE_DIRS}
# ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
# ${Python3_INCLUDE_DIRS})
#
# target_link_libraries(utils_static PUBLIC ${TORCH_LIBRARIES} MPI::MPI_CXX)

# Python Binding (Interface)
add_library(utils_python MODULE ${UTILS_BINDINGS})

target_link_libraries(utils_python PRIVATE Python3::Module ${TORCH_LIBRARIES}
                                           ${TORCH_PYTHON_LIBRARY} MPI::MPI_CXX)

target_include_directories(
  utils_python
  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/bindings/utils
          ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# Set the final .so properties
set_target_properties(
  utils_python
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/ttnte/cpp"
             OUTPUT_NAME "utils"
             PREFIX "")
