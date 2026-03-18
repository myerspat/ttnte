# # Relevant files from src set(UTILS_SRC_REL label.cpp)

# Relevant files from bindings/utils
set(UTILS_BINDINGS_REL bindings.cpp parallel_context.cpp)

# Define the directory paths (Use local variables here, then push if needed)
# set(UTILS_SRC_DIR      "${PROJECT_SOURCE_DIR}/src/utils")
set(UTILS_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/utils")

# Transform the lists (Prepend the path to the relative files) list(TRANSFORM
# UTILS_SRC_REL      PREPEND "${UTILS_SRC_DIR}/")
list(TRANSFORM UTILS_BINDINGS_REL PREPEND "${UTILS_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope set(UTILS_SRC
# ${UTILS_SRC_REL}      PARENT_SCOPE)
set(UTILS_BINDINGS
    ${UTILS_BINDINGS_REL}
    PARENT_SCOPE)
# set(UTILS_SRC_DIR      ${UTILS_SRC_DIR}      PARENT_SCOPE)
set(UTILS_BINDINGS_DIR
    ${UTILS_BINDINGS_DIR}
    PARENT_SCOPE)
