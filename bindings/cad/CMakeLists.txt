# Relevant files from bindings/cad
set(CAD_BINDINGS bindings.cpp bspline_basis.cpp patch.cpp)

# Relevant files from src
set(CAD_SRC ${PROJECT_SOURCE_DIR}/src/cad/bspline_basis.cpp
            ${PROJECT_SOURCE_DIR}/src/cad/patch.cpp)

# Internal C++ Logic (Logic)
add_library(cad_static STATIC ${CAD_SRC})

# We need PIC even for the static lib because it gets swallowed by a shared
# module (.so)
set_target_properties(cad_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  cad_static PUBLIC ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/src ${Python3_INCLUDE_DIRS})

target_link_libraries(cad_static PUBLIC ${TORCH_LIBRARIES} MPI::MPI_CXX)

# Python Binding (Interface)
add_library(cad_python MODULE ${CAD_BINDINGS})

target_link_libraries(
  cad_python PRIVATE cad_static Python3::Module ${TORCH_LIBRARIES}
                     ${TORCH_PYTHON_LIBRARY})

target_include_directories(
  cad_python PRIVATE ${PROJECT_SOURCE_DIR}/bindings/cad ${Python3_INCLUDE_DIRS}
                     ${TORCH_INCLUDE_DIRS})

# Set the final .so properties
set_target_properties(
  cad_python
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/ttnte/cpp"
             OUTPUT_NAME "cad"
             PREFIX "")
