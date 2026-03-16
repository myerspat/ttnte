# Parameters
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

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
	SKBUILD_CMAKE_DEFINE="CMAKE_BUILD_TYPE=$(BUILD_TYPE);TTNTE_OPTIMIZED=$(TTNTE_OPTIMIZED);USE_CUDA=$(USE_CUDA)" \
	$(PIP) install .

dev: BUILD_TYPE ?= Debug
dev: TTNTE_OPTIMIZED ?= OFF
dev: USE_CUDA ?= ON
dev:
	@echo "Installing in $(BUILD_TYPE) mode (Optimized: $(TTNTE_OPTIMIZED), CUDA: $(USE_CUDA))..."
	SKBUILD_CMAKE_DEFINE=$(CMAKE_DEFS) \
	$(PIP) install -e ".[dev]" --no-build-isolation -v

clean:
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +
