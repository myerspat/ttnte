# Define the relative filenames
set(PHYSICS_SRC_REL dg_first_order_transport_backends.cpp)
set(PHYSICS_BINDINGS_REL bindings.cpp boundary_types.cpp assembly_configs.cpp
                         dg_first_order_transport_backends.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(PHYSICS_SRC_DIR "${PROJECT_SOURCE_DIR}/src/physics")
set(PHYSICS_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/physics")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM PHYSICS_SRC_REL PREPEND "${PHYSICS_SRC_DIR}/")
list(TRANSFORM PHYSICS_BINDINGS_REL PREPEND "${PHYSICS_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(PHYSICS_SRC
    ${PHYSICS_SRC_REL}
    PARENT_SCOPE)
set(PHYSICS_BINDINGS
    ${PHYSICS_BINDINGS_REL}
    PARENT_SCOPE)
set(PHYSICS_SRC_DIR
    ${PHYSICS_SRC_DIR}
    PARENT_SCOPE)
set(PHYSICS_BINDINGS_DIR
    ${PHYSICS_BINDINGS_DIR}
    PARENT_SCOPE)
