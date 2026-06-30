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

# ==========================================
# Global Build Configurations
# ==========================================
# Default CUDA to ON globally
USE_CUDA ?= ON

# Dynamically change the defaults if the user runs 'make dev'
ifeq ($(filter dev,$(MAKECMDGOALS)),dev)
	BUILD_TYPE ?= Debug
	TTNTE_OPTIMIZED ?= OFF
else
	BUILD_TYPE ?= Release
	TTNTE_OPTIMIZED ?= ON
endif

.PHONY: help install dev clean

help:
	@echo "Usage:"
	@echo "  make install      - Standard user install"
	@echo "  make dev          - Developer install (build type, editable, no isolation, custom CMake)"
	@echo "  make clean        - Remove build artifacts"

install:
	@echo "Running Production Install (Type: $(BUILD_TYPE), Optimized: $(TTNTE_OPTIMIZED), CUDA: $(USE_CUDA))..."
	CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) \
	SKBUILD_CMAKE_DEFINE="CMAKE_BUILD_TYPE=$(BUILD_TYPE);TTNTE_OPTIMIZED=$(TTNTE_OPTIMIZED);USE_CUDA=$(USE_CUDA)" \
	$(PIP) install $(EDIT_FLAG) . $(VERB_FLAG) --no-build-isolation

dev:
	@echo "Installing in $(BUILD_TYPE) mode (Optimized: $(TTNTE_OPTIMIZED), CUDA: $(USE_CUDA))..."
	CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) \
	SKBUILD_CMAKE_DEFINE="CMAKE_BUILD_TYPE=$(BUILD_TYPE);TTNTE_OPTIMIZED=$(TTNTE_OPTIMIZED);USE_CUDA=$(USE_CUDA)" \
	$(PIP) install -e ".[dev]" --no-build-isolation -v

clean:
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +
