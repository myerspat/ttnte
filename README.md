# ttnte

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`ttnte` is a Python library for solving the discrete ordinates neutron transport equation (NTE) with a discontinuous isogeometric analysis (IGA) spatial discretization. This repository features IGA assembly through PyTorch sparse tensors and tensor trains (TTs). The application of TTs aims to exploit the multiscale structure commonly found in reactor applications. The IGA discretization offers higher continuity than traditional finite elements and benefits from working directly with CAD, cutting out the often expensive meshing step.

## Requirements

- [torch](https://pytorch.org/)
- [torchtt](https://github.com/ion-g-ion/torchTT)
- [numpy](https://numpy.org/)
- [igakit](https://github.com/dalcinl/igakit)
- [geomdl](https://nurbs-python.readthedocs.io/en/5.x/)
- [cotengra](https://cotengra.readthedocs.io/en/latest/index.html)
- [pandas](https://pandas.pydata.org/)

## Installation

Quick installation:
```shall
pip install git+https://github.com/myerspat/ttnte.git
```

Clone the repository:
 
```shell
git clone https://github.com/myerspat/ttnte.git
```

In `ttnte/`, run:

```shell
pip install .
```

## Classes and Methods

- `ttnte.xs.Server`: Class for handling multigroup cross section information.
- `ttnte.iga.IGAMesh`: Meshing object for NURBS surfaces defined as `igakit.nurbs.NURBS`.
- `ttnte.assemblers.MatrixAssembler`: Assembler discretized system into `ttnte.assemblers.operators.SparseOperator`s.
- `ttnte.assemblers.TTAssembler`: Assembler discretized system into `torchtt.TT`s.
- `ttnte.linalg.LinearOperator`: General operator object for `ttnte.assemblers.operators.SparseOperator`s and `torchtt.TT`s.
- `ttnte.linalg.eig()`: Method for solving the resulting discretized eigenvalue problem.

## Modules

- `ttnte.xs.benchmarks`: XS data sets from common neutron transport benchmarks.

## Notebooks

- [Homogeneous square domain](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/square/square.ipynb)
- [Homogeneous circular domain](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/circle/circle.ipynb)
- [Homogeneous quarter circular domain](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/quarter_circle/quarter_circle.ipynb)
- [C5G7 infinite pincell array](https://nbviewer.org/github/myerspat/ttnte/blob/develop/notebooks/pincell/pincell.ipynb)
