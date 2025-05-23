import numpy as np
import torch as tn
from igakit import cad
from run_problem import run_problem

from ttnte.iga import IGAMesh
from ttnte.xs.benchmarks import pu239, research_reactor

tn.set_default_dtype(tn.float64)


def test_quarter_circle():
    # Get XS data
    xs_server = pu239(num_groups=1)

    # Create NURBS surface
    rc = 4.279960  # Critical radius (cm)
    c0 = cad.circle(radius=rc, angle=np.pi / 2)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    patches = {cad.ruled(l0, c0): "Pu-239"}

    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=[3, 9], degree=2)

    # Connect patches
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_condition(("left", "bottom"))

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_problem(mesh, xs_server, 1, 256)
