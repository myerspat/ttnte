import numpy as np
import torch as tn
from igakit import cad
from run_problem import run_problem

from ttnte.iga import IGAMesh
from ttnte.xs.benchmarks import pu239, research_reactor

tn.set_default_dtype(tn.float64)


def test_square_vac():
    # Get XS data
    xs_server = pu239(num_groups=2)

    # Create NURBS geometry
    length = 6.5  # cm
    points = np.array(
        [
            [-length / 2, -length / 2, 0],
            [length / 2, -length / 2, 0],
            [-length / 2, length / 2, 0],
            [length / 2, length / 2, 0],
        ]
    ).reshape((2, 2, -1))
    patches = {cad.bilinear(points): "Pu-239"}

    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=7, degree=3)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_problem(mesh, xs_server, 0.997951, 256)


def test_square_anisotropic():
    # Get XS data
    xs_server = research_reactor(is_anisotropic=True)

    # Create NURBS geometry
    length = 9.4959  # cm
    points = np.array(
        [
            [-length / 2, -length / 2, 0],
            [length / 2, -length / 2, 0],
            [-length / 2, length / 2, 0],
            [length / 2, length / 2, 0],
        ]
    ).reshape((2, 2, -1))
    patches = {cad.bilinear(points): "Research Reactor"}

    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=[5, 7], degree=3)

    # Connect patches
    mesh.connect()

    # Define boundary conditions
    mesh.set_reflective_condition(("left", "top", "bottom"))

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_problem(mesh, xs_server, 1.0, 256)


def test_square_multiregion():
    # Get XS data
    xs_server = research_reactor()

    # Create NURBS geometry
    fuel_length = 6.696802  # cm
    mod_length = 1.126151  # cm

    # Fuel patch
    points = np.array([[-0.5, 0], [0.5, 0], [-0.5, fuel_length], [0.5, fuel_length]])
    patches = {cad.bilinear(points.reshape((2, 2, -1))): "Research Reactor"}

    # Moderator patch
    points[:2, 1] += fuel_length
    points[2:, 1] += mod_length
    patches[cad.bilinear(points.reshape((2, 2, -1)))] = "Water"

    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    for p in range(mesh.num_patches):
        mesh.refine(p, factor=[5, 7], degree=3)

    # Connect patches
    mesh.connect()

    # Define boundary conditions
    mesh.set_reflective_condition(("left", "bottom", "right"))

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_problem(mesh, xs_server, 1.0, 256)
