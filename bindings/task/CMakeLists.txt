# Define the relative filenames
set(TASK_BINDINGS_REL bindings.cpp task.cpp task_graph.cpp task_scheduler.cpp)
set(TASK_SRC_REL task_graph.cpp task_scheduler.cpp)

# Define the directory paths (Use local variables here, then push if needed)
set(TASK_SRC_DIR "${PROJECT_SOURCE_DIR}/src/task")
set(TASK_BINDINGS_DIR "${PROJECT_SOURCE_DIR}/bindings/task")

# Transform the lists (Prepend the path to the relative files)
list(TRANSFORM TASK_SRC_REL PREPEND "${TASK_SRC_DIR}/")
list(TRANSFORM TASK_BINDINGS_REL PREPEND "${TASK_BINDINGS_DIR}/")

# Push the finalized lists and paths to the Parent Scope
set(TASK_SRC
    ${TASK_SRC_REL}
    PARENT_SCOPE)
set(TASK_BINDINGS
    ${TASK_BINDINGS_REL}
    PARENT_SCOPE)
set(TASK_SRC_DIR
    ${TASK_SRC_DIR}
    PARENT_SCOPE)
set(TASK_BINDINGS_DIR
    ${TASK_BINDINGS_DIR}
    PARENT_SCOPE)
