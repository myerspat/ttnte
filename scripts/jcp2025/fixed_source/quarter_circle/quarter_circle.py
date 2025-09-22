import os
import sys
import multiprocessing
from pathlib import Path
from typing import Union, Tuple

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    sys.path.append("../..")

import numpy as np
from igakit import cad

from ttnte.xs.benchmarks import Server
from ttnte.iga import IGAMesh
from ttnte.cad import Patch
from ttnte.sources import IsotropicInternalSource
from ttnte.linalg import LinearSolverOptions, cpp_available

from runner import Runner


def get_xs(num_groups: int):
    """"""
    total = 1  # 1/cm
    scattering_ratio = 0.9
    server = Server(
        {
            "Source": {
                "total": np.array([total]),
                "scatter_gtg": np.array([[[total * scattering_ratio]]]),
            },
            "Void": {
                "total": np.array([0]),
                "scatter_gtg": np.array([[[0]]]),
            },
        }
    )
    assert server.num_groups == num_groups
    return server


def get_mesh(factor: Union[int, Tuple[int]], degree: Union[int, Tuple[int]]):
    """"""
    # Create quarter circle NURBS surface
    radius = 5  # cm
    pitch = 12  # cm
    c0 = cad.circle(radius=radius, angle=np.pi / 2)
    c1 = c0.slice(0, 0, 0.5)
    c2 = c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))

    # Create water patch
    l1 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l2 = cad.line(p0=(pitch / 2, pitch / 2), p1=(0, pitch / 2))

    # Create NURBS surfaces
    source = [Patch(cad.ruled(l0, c1), "Source"), Patch(cad.ruled(l0, c2), "Source")]
    void = [Patch(cad.ruled(c1, l1), "Void"), Patch(cad.ruled(c2, l2), "Void")]

    # Add uniform source of 1/cm to patch
    source[0].set_source(IsotropicInternalSource(np.ones((1, *source[0].shape))))
    source[1].set_source(IsotropicInternalSource(np.ones((1, *source[1].shape))))

    # Initialize IGA mesh and add the patches
    mesh = IGAMesh(max_processes=32)
    for patch in source + void:
        mesh.add_patch(patch)

    # Refine each patch to have 6 knot spans with degree 2
    mesh.refine(factor=factor, degree=degree)

    # Connect patches
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_conditions(("left", "bottom"))

    # Finalize mesh
    mesh.finalize()
    print(mesh)
    return mesh
