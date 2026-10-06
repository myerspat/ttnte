# Define the relative filenames
set(SWEEP_BINDINGS_REL bindings.cpp sweep_config.cpp patch_discretization.cpp
                       sweep_graph.cpp sweep_driver.cpp)
set(SWEEP_SRC_REL angular_moments.cpp patch_discretization.cpp sweep_graph.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(SWEEP_SRC_DIR "${PROJECT_SOURCE_DIR}/src/sweep")
set(SWEEP_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/sweep")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM SWEEP_SRC_REL PREPEND "${SWEEP_SRC_DIR}/")
list(TRANSFORM SWEEP_BINDINGS_REL PREPEND "${SWEEP_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(SWEEP_SRC
    ${SWEEP_SRC_REL}
    PARENT_SCOPE)
set(SWEEP_BINDINGS
    ${SWEEP_BINDINGS_REL}
    PARENT_SCOPE)
set(SWEEP_SRC_DIR
    ${SWEEP_SRC_DIR}
    PARENT_SCOPE)
set(SWEEP_BINDINGS_DIR
    ${SWEEP_BINDINGS_DIR}
    PARENT_SCOPE)
