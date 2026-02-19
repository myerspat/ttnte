# Relevant files from bindings/cad
set(CAD_BINDINGS
    bindings.cpp
    utils.cpp
    basis_functions.cpp
    )


# Relevant files from src
set(CAD_SRC
    ${PROJECT_SOURCE_DIR}/src/cad/basis_functions.cpp
    )



add_library(cad MODULE ${CAD_BINDINGS} ${CAD_SRC})

target_include_directories(
  cad
  PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/bindings/cad
          ${Python3_INCLUDE_DIR})

# Link libraries
target_link_libraries(cad PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# Set the ABI flag to match PyTorch
target_compile_definitions(cad PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}
)

# Set output name
set_target_properties(
  cad
  PROPERTIES OUTPUT_NAME "cad"
             PREFIX "cpp/"
             SUFFIX ".so")
