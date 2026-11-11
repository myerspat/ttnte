# Define the relative filenames
set(TENSOR_BINDINGS_REL bindings.cpp index.cpp layout.cpp)
set(TENSOR_SRC_REL index.cpp layout.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(TENSOR_SRC_DIR "${PROJECT_SOURCE_DIR}/src/tensor")
set(TENSOR_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/tensor")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM TENSOR_SRC_REL PREPEND "${TENSOR_SRC_DIR}/")
list(TRANSFORM TENSOR_BINDINGS_REL PREPEND "${TENSOR_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(TENSOR_SRC
    ${TENSOR_SRC_REL}
    PARENT_SCOPE)
set(TENSOR_BINDINGS
    ${TENSOR_BINDINGS_REL}
    PARENT_SCOPE)
set(TENSOR_SRC_DIR
    ${TENSOR_SRC_DIR}
    PARENT_SCOPE)
set(TENSOR_BINDINGS_DIR
    ${TENSOR_BINDINGS_DIR}
    PARENT_SCOPE)
