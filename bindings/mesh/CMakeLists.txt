# Define the relative filenames
set(MESH_BINDINGS_REL bindings.cpp mesh_block_boundary.cpp)
# set(MESH_SRC_REL mesh.cpp)

# Define the directory paths (Use local variables here, then push if needed)
# set(MESH_SRC_DIR "${PROJECT_SOURCE_DIR}/src/mesh")
set(MESH_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/mesh")

# Transform the lists (Prepend the path to the relative files) list(TRANSFORM
# MESH_SRC_REL PREPEND "${MESH_SRC_DIR}/")
list(TRANSFORM MESH_BINDINGS_REL PREPEND "${MESH_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope set(MESH_SRC
# ${MESH_SRC_REL} PARENT_SCOPE)
set(MESH_BINDINGS
    ${MESH_BINDINGS_REL}
    PARENT_SCOPE)
# set(MESH_SRC_DIR ${MESH_SRC_DIR} PARENT_SCOPE)
set(MESH_BINDINGS_DIR
    ${MESH_BINDINGS_DIR}
    PARENT_SCOPE)
