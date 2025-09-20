# Relevant files from bindings/linalg
set(LINALG_BINDINGS
    bindings.cpp
    tt_operator.cpp
    contraction_step.cpp
    sparse_operator.cpp
    scatter_operator.cpp
    fission_operator.cpp
    operator.cpp
    linear_operator.cpp
    gmres.cpp
    utils.cpp
    eig.cpp)

# Relevant files from src
set(LINALG_SRC
    ${PROJECT_SOURCE_DIR}/src/linalg/tt_operator.cpp
    ${PROJECT_SOURCE_DIR}/src/linalg/contraction_step.cpp
    ${PROJECT_SOURCE_DIR}/src/linalg/sparse_operator.cpp
    ${PROJECT_SOURCE_DIR}/src/linalg/scatter_operator.cpp
    ${PROJECT_SOURCE_DIR}/src/linalg/fission_operator.cpp
    ${PROJECT_SOURCE_DIR}/src/linalg/operator.cpp
    ${PROJECT_SOURCE_DIR}/src/linalg/linear_operator.cpp
    ${PROJECT_SOURCE_DIR}/src/linalg/gmres.cpp
    ${PROJECT_SOURCE_DIR}/src/linalg/eig.cpp)

add_library(linalg MODULE ${LINALG_BINDINGS} ${LINALG_SRC})
target_include_directories(
  linalg
  PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/bindings/linalg
          ${Python3_INCLUDE_DIR})

# Link libraries
target_link_libraries(linalg PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# Set output name
set_target_properties(
  linalg
  PROPERTIES OUTPUT_NAME "linalg"
             PREFIX "cpp/"
             SUFFIX ".so")
