# Relevant files from src
set(PY_SRC_REL package_manager.cpp acquire.cpp release.cpp torchtt.cpp
               numpy.cpp)

# Relevant files from bindings/python
set(PY_BINDINGS_REL bindings.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(PY_SRC_DIR "${PROJECT_SOURCE_DIR}/src/python")
set(PY_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/python")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM PY_SRC_REL PREPEND "${PY_SRC_DIR}/")
list(TRANSFORM PY_BINDINGS_REL PREPEND "${PY_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(PY_SRC
    ${PY_SRC_REL}
    PARENT_SCOPE)
set(PY_BINDINGS
    ${PY_BINDINGS_REL}
    PARENT_SCOPE)
set(PY_SRC_DIR
    ${PY_SRC_DIR}
    PARENT_SCOPE)
set(PY_BINDINGS_DIR
    ${PY_BINDINGS_DIR}
    PARENT_SCOPE)
