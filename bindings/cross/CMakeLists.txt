# Define the relative filenames
set(CROSS_BINDINGS_REL bindings.cpp cross_config.cpp layout_cross.cpp
                       pivots.cpp)
set(CROSS_SRC_REL
    pivots.cpp layout_cross.cpp layout_cross_check.cpp
    layout_cross_contract.cpp layout_cross_fibers.cpp layout_cross_select.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(CROSS_SRC_DIR "${PROJECT_SOURCE_DIR}/src/cross")
set(CROSS_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/cross")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM CROSS_SRC_REL PREPEND "${CROSS_SRC_DIR}/")
list(TRANSFORM CROSS_BINDINGS_REL PREPEND "${CROSS_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(CROSS_SRC
    ${CROSS_SRC_REL}
    PARENT_SCOPE)
set(CROSS_BINDINGS
    ${CROSS_BINDINGS_REL}
    PARENT_SCOPE)
set(CROSS_SRC_DIR
    ${CROSS_SRC_DIR}
    PARENT_SCOPE)
set(CROSS_BINDINGS_DIR
    ${CROSS_BINDINGS_DIR}
    PARENT_SCOPE)
