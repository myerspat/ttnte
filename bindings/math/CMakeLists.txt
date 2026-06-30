# Define the relative filenames
set(MATH_BINDINGS_REL bindings.cpp special.cpp quadrature_set.cpp)
set(MATH_SRC_REL special.cpp quadrature_set.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(MATH_SRC_DIR "${PROJECT_SOURCE_DIR}/src/math")
set(MATH_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/math")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM MATH_SRC_REL PREPEND "${MATH_SRC_DIR}/")
list(TRANSFORM MATH_BINDINGS_REL PREPEND "${MATH_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(MATH_SRC
    ${MATH_SRC_REL}
    PARENT_SCOPE)
set(MATH_BINDINGS
    ${MATH_BINDINGS_REL}
    PARENT_SCOPE)
set(MATH_SRC_DIR
    ${MATH_SRC_DIR}
    PARENT_SCOPE)
set(MATH_BINDINGS_DIR
    ${MATH_BINDINGS_DIR}
    PARENT_SCOPE)
