# # Relevant files from src
set(LINALG_SRC_REL
    linear_system.cpp
    operator.cpp
    state.cpp
    tt_engine.cpp
    tt_linear_system.cpp
    tt_operator.cpp
    tt_state.cpp)

# Relevant files from bindings/linalg
set(LINALG_BINDINGS_REL
    bindings.cpp
    operator.cpp
    state.cpp
    tt_engine.cpp
    tt_operator.cpp
    tt_state.cpp
    linear_system.cpp
    tt_linear_system.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(LINALG_SRC_DIR "${PROJECT_SOURCE_DIR}/src/linalg")
set(LINALG_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/linalg")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM LINALG_SRC_REL PREPEND "${LINALG_SRC_DIR}/")
list(TRANSFORM LINALG_BINDINGS_REL PREPEND "${LINALG_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(LINALG_SRC
    ${LINALG_SRC_REL}
    PARENT_SCOPE)
set(LINALG_BINDINGS
    ${LINALG_BINDINGS_REL}
    PARENT_SCOPE)
set(LINALG_SRC_DIR
    ${LINALG_SRC_DIR}
    PARENT_SCOPE)
set(LINALG_BINDINGS_DIR
    ${LINALG_BINDINGS_DIR}
    PARENT_SCOPE)
