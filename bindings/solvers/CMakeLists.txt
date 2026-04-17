# Define the relative filenames
set(SOLVERS_BINDINGS_REL
    bindings.cpp dd_strategy.cpp memory_policy.cpp block_jacobi_strategy.cpp
    local_solver.cpp amen_solver.cpp)
set(SOLVERS_SRC_REL block_jacobi_strategy.cpp dd_strategy.cpp amen_solver.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(SOLVERS_SRC_DIR "${PROJECT_SOURCE_DIR}/src/solvers")
set(SOLVERS_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/solvers")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM SOLVERS_SRC_REL PREPEND "${SOLVERS_SRC_DIR}/")
list(TRANSFORM SOLVERS_BINDINGS_REL PREPEND "${SOLVERS_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(SOLVERS_SRC
    ${SOLVERS_SRC_REL}
    PARENT_SCOPE)
set(SOLVERS_BINDINGS
    ${SOLVERS_BINDINGS_REL}
    PARENT_SCOPE)
set(SOLVERS_SRC_DIR
    ${SOLVERS_SRC_DIR}
    PARENT_SCOPE)
set(SOLVERS_BINDINGS_DIR
    ${SOLVERS_BINDINGS_DIR}
    PARENT_SCOPE)
