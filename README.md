# ttnte

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests Status](https://github.com/myerspat/ttnte/actions/workflows/CI.yml/badge.svg)](https://github.com/myerspat/ttnte/actions/workflows)

`ttnte` is a Python library for solving the discrete ordinates neutron transport
equation (NTE) with a discontinuous isogeometric analysis (IGA) spatial
discretization. This repository features IGA assembly through PyTorch sparse
tensors and tensor trains (TTs). The application of TTs aims to exploit the
multiscale structure commonly found in reactor applications. The IGA
discretization offers higher continuity than traditional finite elements and
benefits from working directly with CAD, cutting out the often expensive meshing
step.

## Requirements

- [torch](https://pytorch.org/)
- [torchtt](https://github.com/ion-g-ion/torchTT)
- [numpy](https://numpy.org/)
- [igakit](https://github.com/dalcinl/igakit)
- [geomdl](https://nurbs-python.readthedocs.io/en/5.x/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [plotly](https://plotly.com/python/)
- [h5py](https://docs.h5py.org/en/stable/index.html)
- [tqdm](https://github.com/tqdm/tqdm)
- [cotengra](https://cotengra.readthedocs.io/en/latest/index.html)
- [optuna](https://optuna.org/)
- [cmaes](https://pypi.org/project/cmaes/)
- [cotengrust](https://github.com/jcmgray/cotengrust)
- [cytools](https://cy.tools/)
- [kahypar](https://kahypar.org/)
- [loky](https://loky.readthedocs.io/en/stable/index.html)
- [networkx](https://networkx.org/)
- [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)

> [!NOTE] For best performance compile
> [geomdl with Cython](https://nurbs-python.readthedocs.io/en/5.x/install.html#compile-with-cython),
> [PyTorch with CUDA](https://pytorch.org/get-started/locally/),
> [torchTT with its C++ extension](https://github.com/ion-g-ion/torchTT), and
> ensure `ttnte` compiles by setting `TTNTE_CPP_BACKEND=True` environment
> variable. `ttnte` will compile the C++ backend by default but if it fails to
> compile it will fall back on the Python implementations.

## Installation

### Quick installation

```shell
pip install https://github.com/dalcinl/igakit/archive/refs/heads/master.zip
pip install git+https://github.com/ion-g-ion/torchTT.git
pip install git+https://github.com/myerspat/ttnte.git
```

### From source

```shell
git clone https://github.com/myerspat/ttnte.git && cd ttnte
pip install https://github.com/dalcinl/igakit/archive/refs/heads/master.zip
pip install git+https://github.com/ion-g-ion/torchTT.git
pip install .
```

### For developers

```shell
git clone git@github.com:myerspat/ttnte.git && cd ttnte
pip install https://github.com/dalcinl/igakit/archive/refs/heads/master.zip
pip install git+https://github.com/ion-g-ion/torchTT.git
pip install -e ".[dev]"
pre-commit install
```

### Install Environment Variables

- `TTNTE_CPP_BACKEND`: Ensure the C++ backend compiles by setting this to
  `True`. You can also toggle the C++ backend at runtime.
- `DEBUG`: Compile the C++ backend in debug mode.
- `NJOBS`: Number of threads to use when compiling. This defaults to
  `os.cpu_count()`.
- `TTNTE_PROFILE`: By default if `DEBUG=True` this environment variable defaults
  to `True`. This toggles compilation with `-g` and `-fno-omit-frame-pointer`
  flags.
- `TTNTE_OPTIMIZED`: If this is set to `True` then the C++ backend is compiled
  with `-O3` and `-march=native` flags. This defaults to `True` unless
  `DEBUG=True`.
- `USE_CUDA`: Link PyTorch with CUDA support. This defaults to `True`.
- `TORCH_INSTALL_PREFIX`: Path to the base PyTorch directory. This defaults to
  `os.path.abspath(os.path.dirname(torch.__file__))`.
- `_GLIBCXX_USE_CXX_ABI`: This is either `0` or `1` and is used by PyTorch. This
  defaults to `int(torch._C._GLIBCXX_USE_CXX11_ABI)`.

## Classes and Methods

- `ttnte.xs.Server`: Class for handling multigroup cross section information.
- `ttnte.cad.Patch`: Patch class.
- `ttnte.iga.IGAMesh`: Meshing object for NURBS surfaces defined as
  `igakit.nurbs.NURBS`.
- `ttnte.assemblers.MatrixAssembler`: Assembler discretized system into
  `ttnte.assemblers.operators.SparseOperator`s.
- `ttnte.assemblers.TTAssembler`: Assembler discretized system into
  `torchtt.TT`s.
- `ttnte.linalg.gmres()`: Method for solving the resulting discretized fixed
  source systems
- `ttnte.linalg.power()`: Method for solving the resulting discretized
  eigenvalue problem.
- `ttnte.linalg.LinearSolverOptions`: Options class for GMRES used in
  `ttnte.linalg.power()`.

## Modules

- `ttnte.xs.benchmarks`: XS data sets from common neutron transport benchmarks.
- `ttnte.cad.curves`: Methods for building NURBS curves used in the notebooks.
- `ttnte.cad.surfaces`: Methods for building NURBS surfaces used in the
  notebooks.
- `ttnte.sources`: Define fixed sources.

## Notebooks

### Fixed Source

- [Homogeneous square](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/fixed_source/square/square.ipynb)
- [Homogeneous circle](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/fixed_source/circle/circle.ipynb)
- [Quarter circle with void](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/fixed_source/quarter_circle/quarter_circle.ipynb)
- [Cruciform source with cylindrical wall](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/fixed_source/cruciform/cruciform.ipynb)

### Eigenvalue

- [Homogeneous square](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/eigenvalue/square/square.ipynb)
- [Homogeneous circle](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/eigenvalue/circle/circle.ipynb)
- [Homogeneous quarter circle](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/eigenvalue/quarter_circle/quarter_circle.ipynb)
- [C5G7 infinite pincell array](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/eigenvalue/pincell/pincell.ipynb)
- Infinite array of Lightbridge four-lobe fuel with a
  - [burnable absorder displacer](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/eigenvalue/lightbridge/lightbridge_ba.ipynb)
  - [gas displacer](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/eigenvalue/lightbridge/lightbridge_gas.ipynb)
