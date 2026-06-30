# Relevant files from src
set(XS_SRC_REL material.cpp server.cpp)

# Relevant files from bindings/xs
set(XS_BINDINGS_REL bindings.cpp material.cpp server.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(XS_SRC_DIR "${PROJECT_SOURCE_DIR}/src/xs")
set(XS_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/xs")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM XS_SRC_REL PREPEND "${XS_SRC_DIR}/")
list(TRANSFORM XS_BINDINGS_REL PREPEND "${XS_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(XS_SRC
    ${XS_SRC_REL}
    PARENT_SCOPE)
set(XS_BINDINGS
    ${XS_BINDINGS_REL}
    PARENT_SCOPE)
set(XS_SRC_DIR
    ${XS_SRC_DIR}
    PARENT_SCOPE)
set(XS_BINDINGS_DIR
    ${XS_BINDINGS_DIR}
    PARENT_SCOPE)
