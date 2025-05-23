import numpy as np
import torch as tn
from igakit.nurbs import NURBS
from run_problem import run_problem

from ttnte.iga import IGAMesh
from ttnte.xs.benchmarks import pu239, research_reactor

tn.set_default_dtype(tn.float64)


def circle(radius):
    """
    Create circular NURBS surface with given radius.

    Parameters
    ----------
    radius: float
        Radius of the circle.

    Returns
    -------
    circle: igakit.nurbs.NURBS
        NURBS surface.
    """
    # Control points and weights
    ctrlpts = radius * np.array(
        [
            [-1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, -np.sqrt(2), 0],
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [-np.sqrt(2), 0, 0],
            [0, 0, 0],
            [np.sqrt(2), 0, 0],
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, np.sqrt(2), 0],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        ]
    ).reshape((3, 3, 3))
    weights = np.array(
        [
            [1, 1 / np.sqrt(2), 1],
            [1 / np.sqrt(2), 1, 1 / np.sqrt(2)],
            [1, 1 / np.sqrt(2), 1],
        ]
    )

    return NURBS(
        knots=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], control=ctrlpts, weights=weights
    )


def test_square():
    # Get XS data
    xs_server = pu239(num_groups=1)

    # Create NURBS surface
    rc = 4.279960  # Critical radius (cm)
    patches = {circle(rc): "Pu-239"}

    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=7, degree=2)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_problem(mesh, xs_server, 1, 256)
