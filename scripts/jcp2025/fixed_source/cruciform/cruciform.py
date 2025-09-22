import os
import sys
import multiprocessing
from pathlib import Path
from typing import Union, Tuple

multiprocessing.set_start_method("forkserver")
sys.path.append("../..")

import numpy as np
from igakit import cad

from ttnte.xs.benchmarks import Server
from ttnte.iga import IGAMesh
from ttnte.cad import Patch
from ttnte.cad.curves import qtrlobe
from ttnte.sources import IsotropicInternalSource
from ttnte.linalg import LinearSolverOptions, cpp_available

from runner import Runner


def get_xs(num_groups: int):
    """"""
    server = Server(
        {
            "Source": {
                "total": np.array([0.01]),
                "scatter_gtg": np.array([[[0.008]]]),
            },
            "Void": {
                "total": np.array([0]),
                "scatter_gtg": np.array([[[0]]]),
            },
            "Shield": {
                "total": np.array([3]),
                "scatter_gtg": np.array([[[0.5]]]),
            },
        }
    )

    assert server.num_groups == num_groups
    return server


def get_mesh(factor: Union[int, Tuple[int]], degree: Union[int, Tuple[int]]):
    """"""
    # Initialize dimensional variables
    X = 10  # Channel pitch

    # Cruciform
    R = 2  # Radius defining valleys of fixed source
    delta = 1  # Width of lobes
    d2 = delta * 0.5  # Half width of lobes
    x = 0.25  # Portrusion of lobes

    # Shielding
    I = 3.75  # Inner radius
    O = 4.5  # Outer radius

    # NURBS curves
    origin = cad.line(p0=(0, 0), p1=(0, 0))
    cruciform = qtrlobe(outrad=R, portrs=x, hfwidth=d2)
    circleI = cad.circle(radius=I, angle=[np.pi / 2, 0])
    circleO = cad.circle(radius=O, angle=[np.pi / 2, 0])
    topedge = cad.line(p0=(0, X / 2), p1=(X / 2, X / 2))
    corner = cad.line(p1=(X / 2, X / 2), p0=(X / 2, X / 2))
    rightedge = cad.line(p1=(X / 2, 0), p0=(X / 2, X / 2))

    # Create IGA mesh object
    mesh = IGAMesh(max_processes=32)

    # Create and add NURBS surfaces
    sections = [0, 1 / 3, 2 / 3, 1]
    edges = [topedge, corner, rightedge]

    for i in range(len(sections) - 1):
        # Line sections
        csec = origin.slice(0, sections[i], sections[i + 1])
        ssec = cruciform.slice(0, sections[i], sections[i + 1])
        isec = circleI.slice(0, sections[i], sections[i + 1])
        osec = circleO.slice(0, sections[i], sections[i + 1])

        # Create source patch
        source = Patch(cad.ruled(csec, ssec), "Source")
        source.set_source(IsotropicInternalSource(np.ones((1, *source.shape))))
        mesh.add_patch(source)

        # Add remaining
        mesh.add_patch(Patch(cad.ruled(ssec, isec), "Void"))
        mesh.add_patch(Patch(cad.ruled(isec, osec), "Shield"))
        mesh.add_patch(Patch(cad.ruled(osec, edges[i]), "Void"))

    # Refine mesh resolution
    mesh.refine(factor=factor, degree=degree)

    # Connect patches
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_conditions(("left", "bottom"))

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
    eps = [1e-3, 1e-5, 1e-8]

    # Define GMRES configuration
    lsoptions = LinearSolverOptions()

    # MC solutions
    mc_leakage_frac = [0.06913173400000001, 1.1401809264552177e-05]
    mc_phi = np.load(
        dir / "../../../../notebooks/fixed_source/cruciform/openmc/data/mesh_flux.npy"
    )
    mc_phi_stdev = np.load(
        dir / "../../../../notebooks/fixed_source/cruciform/openmc/data/mesh_stdev.npy"
    )

    # =======================================================
    # Direction scaling study
    num_ordinates = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]
    factors = [10]

    # Create runner
    runner = Runner(
        study_name="direction",
        study_path=os.path.dirname(os.path.abspath(__file__)),
        num_ordinates=num_ordinates,
        num_groups=num_groups,
        factors=factors,
        degrees=degrees,
        eps=eps,
        gpu_idx=0,
        cpu_and_gpu=True,
        verbose=True,
        mc_leakage_fraction=mc_leakage_frac,
        mc_solution=[mc_phi, mc_phi_stdev],
    )

    # Run problems
    runner.run(
        get_xs=get_xs,
        get_mesh=get_mesh,
        lsoptions=lsoptions,
    )

    # # =======================================================
    # # Mesh size scaling study
    # num_ordinates = [256]
