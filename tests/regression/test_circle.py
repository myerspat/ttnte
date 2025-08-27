import multiprocessing

multiprocessing.set_start_method("spawn")

import numpy as np
import torch as tn
from igakit.nurbs import NURBS
from run_problem import run_eig

from ttnte.cad import Patch
from ttnte.cad.surfaces import circle
from ttnte.iga import IGAMesh
from ttnte.xs.benchmarks import pu239, research_reactor

tn.set_default_dtype(tn.float64)


def test_square():
    # Get XS data
    xs_server = pu239(num_groups=1)

    # Create mesh
    mesh = IGAMesh()

    # Create NURBS surface
    rc = 4.279960  # Critical radius (cm)
    mesh.add_patch(Patch(circle(rc), "Pu-239"))

    # Refine mesh resolution
    mesh.refine(factor=7, degree=2)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_eig(mesh, xs_server, 1, 256)
