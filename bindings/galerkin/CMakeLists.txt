# Define the relative filenames
set(GALERKIN_BINDINGS_REL bindings.cpp galerkin_config.cpp greedy_galerkin.cpp)
set(GALERKIN_SRC_REL greedy_galerkin.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(GALERKIN_SRC_DIR "${PROJECT_SOURCE_DIR}/src/galerkin")
set(GALERKIN_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/galerkin")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM GALERKIN_SRC_REL PREPEND "${GALERKIN_SRC_DIR}/")
list(TRANSFORM GALERKIN_BINDINGS_REL PREPEND "${GALERKIN_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(GALERKIN_SRC
    ${GALERKIN_SRC_REL}
    PARENT_SCOPE)
set(GALERKIN_BINDINGS
    ${GALERKIN_BINDINGS_REL}
    PARENT_SCOPE)
set(GALERKIN_SRC_DIR
    ${GALERKIN_SRC_DIR}
    PARENT_SCOPE)
set(GALERKIN_BINDINGS_DIR
    ${GALERKIN_BINDINGS_DIR}
    PARENT_SCOPE)
