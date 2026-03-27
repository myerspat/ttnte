# Define the relative filenames
set(DRIVER_BINDINGS_REL bindings.cpp)
# set(DRIVER_SRC_REL driver.cpp)

# Define the directory paths (Use local variables here, then push if needed)
# set(DRIVER_SRC_DIR "${PROJECT_SOURCE_DIR}/src/driver")
set(DRIVER_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/driver")

# Transform the lists (Prepend the path to the relative files) list(TRANSFORM
# DRIVER_SRC_REL PREPEND "${DRIVER_SRC_DIR}/")
list(TRANSFORM DRIVER_BINDINGS_REL PREPEND "${DRIVER_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope set(DRIVER_SRC
# ${DRIVER_SRC_REL} PARENT_SCOPE)
set(DRIVER_BINDINGS
    ${DRIVER_BINDINGS_REL}
    PARENT_SCOPE)
# set(DRIVER_SRC_DIR ${DRIVER_SRC_DIR} PARENT_SCOPE)
set(DRIVER_BINDINGS_DIR
    ${DRIVER_BINDINGS_DIR}
    PARENT_SCOPE)
