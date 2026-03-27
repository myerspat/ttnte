# Define the relative filenames
set(PARALLEL_SRC_REL communicator.cpp parallel_context.cpp request.cpp
                     parmetis_ops.cpp)
set(PARALLEL_BINDINGS_REL bindings.cpp parallel_context.cpp routing_table.cpp
                          communicator.cpp request.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(PARALLEL_SRC_DIR "${PROJECT_SOURCE_DIR}/src/parallel")
set(PARALLEL_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/parallel")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM PARALLEL_SRC_REL PREPEND "${PARALLEL_SRC_DIR}/")
list(TRANSFORM PARALLEL_BINDINGS_REL PREPEND "${PARALLEL_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(PARALLEL_SRC
    ${PARALLEL_SRC_REL}
    PARENT_SCOPE)
set(PARALLEL_BINDINGS
    ${PARALLEL_BINDINGS_REL}
    PARENT_SCOPE)
set(PARALLEL_SRC_DIR
    ${PARALLEL_SRC_DIR}
    PARENT_SCOPE)
set(PARALLEL_BINDINGS_DIR
    ${PARALLEL_BINDINGS_DIR}
    PARENT_SCOPE)
