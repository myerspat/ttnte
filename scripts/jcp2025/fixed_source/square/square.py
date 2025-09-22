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
            }
        }
    )
    assert server.num_groups == num_groups
    return server


def get_mesh(factor: Union[int, Tuple[int]], degree: Union[int, Tuple[int]]):
    """"""
    # Create NURBS geometry
    length = 10  # cm
    points = np.array(
        [
            [-length / 2, -length / 2, 0],
            [length / 2, -length / 2, 0],
            [-length / 2, length / 2, 0],
            [length / 2, length / 2, 0],
        ]
    ).reshape((2, 2, -1))
    patch = Patch(cad.bilinear(points), "Source")

    # Add uniform source of 1/cm to patch
    source = IsotropicInternalSource(np.ones((1, *patch.shape)))
    patch.set_source(source)

    # Create mesh
    mesh = IGAMesh()
    mesh.add_patch(patch)

    # Refine mesh resolution
    mesh.refine(factor=factor, degree=degree)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()
    print(mesh)
    return mesh


if __name__ == "__main__":
    if cpp_available == False:
        raise RuntimeError("C++ backend was not configured")

    # Path to this directory
    dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Combinations to run
    num_groups = [1]
    degrees = [2, 3, 6]
    eps = [1e-8, 1e-5, 1e-3]

    # Define GMRES configuration
    lsoptions = LinearSolverOptions(
        tol=1e-6, maxiter=1000, restart=100, solve_method="batched", verbose=True
    )

    # =======================================================
    # Direction scaling study
    num_ordinates = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]
    factors = [10]

    # Create runner
    runner = Runner(
        study_name="direction",
        study_path=dir / "direction",
        num_ordinates=num_ordinates,
        num_groups=num_groups,
        factors=factors,
        degrees=degrees,
        eps=eps,
        gpu_idx=0,
        cpu_and_gpu=True,
        verbose=True,
    )

    # Run problems
    runner.run(
        get_xs=get_xs,
        get_mesh=get_mesh,
        lsoptions=lsoptions,
    )

    # =======================================================
    # Mesh scaling study
    num_ordinates = [256]
    factors = np.geomspace(5, 10, 8).astype(int).tolist()

    # Create runner
    runner = Runner(
        study_name="meshsize",
        study_path=dir / "meshsize",
        num_ordinates=num_ordinates,
        num_groups=num_groups,
        factors=factors,
        degrees=degrees,
        eps=eps,
        gpu_idx=0,
        cpu_and_gpu=True,
        verbose=True,
    )

    # Run problems
    runner.run(
        get_xs=get_xs,
        get_mesh=get_mesh,
        lsoptions=lsoptions,
    )
