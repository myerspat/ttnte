# Parameters
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

# Default to 4 threads, but allow overriding (e.g., make dev JOBS=8)
JOBS ?= 4

# Optional pip toggles (Set to 1 to enable)
EDITABLE ?= OFF
VERBOSE ?= OFF

# Make translates the toggles into actual pip flags
EDIT_FLAG = $(if $(filter ON,$(EDITABLE)),-e,)
VERB_FLAG = $(if $(filter ON,$(VERBOSE)),-v,)

.PHONY: help install dev clean

help:
	@echo "Usage:"
	@echo "  make install      - Standard user install"
	@echo "  make dev          - Developer install (build type, editable, no isolation, custom CMake)"
	@echo "  make clean        - Remove build artifacts"

install: BUILD_TYPE ?= Release
install: TTNTE_OPTIMIZED ?= ON
install: USE_CUDA ?= ON
install:
	@echo "Running Production Install..."
	CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) \
	SKBUILD_CMAKE_DEFINE="CMAKE_BUILD_TYPE=$(BUILD_TYPE);TTNTE_OPTIMIZED=$(TTNTE_OPTIMIZED);USE_CUDA=$(USE_CUDA)" \
	$(PIP) install $(EDIT_FLAG) . $(VERB_FLAG) --no-build-isolation

dev: BUILD_TYPE ?= Debug
dev: TTNTE_OPTIMIZED ?= OFF
dev: USE_CUDA ?= ON
dev:
	@echo "Installing in $(BUILD_TYPE) mode (Optimized: $(TTNTE_OPTIMIZED), CUDA: $(USE_CUDA))..."
	CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) \
	SKBUILD_CMAKE_DEFINE="CMAKE_BUILD_TYPE=$(BUILD_TYPE);TTNTE_OPTIMIZED=$(TTNTE_OPTIMIZED);USE_CUDA=$(USE_CUDA)" \
	$(PIP) install -e ".[dev]" --no-build-isolation -v

clean:
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +
