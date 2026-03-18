# Define the relative filenames
set(CAD_BINDINGS_REL bindings.cpp bspline_basis.cpp patch.cpp)
set(CAD_SRC_REL bspline_basis.cpp patch.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(CAD_SRC_DIR "${PROJECT_SOURCE_DIR}/src/cad")
set(CAD_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/cad")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM CAD_SRC_REL PREPEND "${CAD_SRC_DIR}/")
list(TRANSFORM CAD_BINDINGS_REL PREPEND "${CAD_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(CAD_SRC
    ${CAD_SRC_REL}
    PARENT_SCOPE)
set(CAD_BINDINGS
    ${CAD_BINDINGS_REL}
    PARENT_SCOPE)
set(CAD_SRC_DIR
    ${CAD_SRC_DIR}
    PARENT_SCOPE)
set(CAD_BINDINGS_DIR
    ${CAD_BINDINGS_DIR}
    PARENT_SCOPE)
