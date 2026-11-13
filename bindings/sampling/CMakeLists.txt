# Define the relative filenames
set(SAMPLING_BINDINGS_REL bindings.cpp weighting.cpp)
set(SAMPLING_SRC_REL fiber_store.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(SAMPLING_SRC_DIR "${PROJECT_SOURCE_DIR}/src/sampling")
set(SAMPLING_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/sampling")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM SAMPLING_SRC_REL PREPEND "${SAMPLING_SRC_DIR}/")
list(TRANSFORM SAMPLING_BINDINGS_REL PREPEND "${SAMPLING_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(SAMPLING_SRC
    ${SAMPLING_SRC_REL}
    PARENT_SCOPE)
set(SAMPLING_BINDINGS
    ${SAMPLING_BINDINGS_REL}
    PARENT_SCOPE)
set(SAMPLING_SRC_DIR
    ${SAMPLING_SRC_DIR}
    PARENT_SCOPE)
set(SAMPLING_BINDINGS_DIR
    ${SAMPLING_BINDINGS_DIR}
    PARENT_SCOPE)
