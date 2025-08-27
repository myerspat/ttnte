import numpy as np
import torch as tn
from igakit import cad
from run_problem import run_eig, run_fixed_source

from ttnte.cad import Patch
from ttnte.iga import IGAMesh
from ttnte.sources import IsotropicInternalSource
from ttnte.xs import Server
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

    # Create mesh
    mesh = IGAMesh()
    mesh.add_patch(Patch(cad.bilinear(points), "Pu-239"))

    # Refine mesh resolution
    mesh.refine(factor=7, degree=3)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_eig(mesh, xs_server, 0.997951, 256)


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

    # Create mesh
    mesh = IGAMesh()
    mesh.add_patch(Patch(cad.bilinear(points), "Research Reactor"))

    # Refine mesh resolution
    mesh.refine(factor=[5, 7], degree=3)

    # Connect patches
    mesh.connect()

    # Define boundary conditions
    mesh.set_reflective_conditions(("left", "top", "bottom"))

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_eig(mesh, xs_server, 1.0, 256)


def test_square_multiregion():
    # Get XS data
    xs_server = research_reactor()

    # Create NURBS geometry
    fuel_length = 6.696802  # cm
    mod_length = 1.126151  # cm

    # Initilize IGA mesh
    mesh = IGAMesh()

    # Fuel patch
    points = np.array([[-0.5, 0], [0.5, 0], [-0.5, fuel_length], [0.5, fuel_length]])
    mesh.add_patch(Patch(cad.bilinear(points.reshape((2, 2, -1))), "Research Reactor"))

    # Moderator patch
    points[:2, 1] += fuel_length
    points[2:, 1] += mod_length
    mesh.add_patch(Patch(cad.bilinear(points.reshape((2, 2, -1))), "Water"))

    # Refine mesh resolution
    mesh.refine(factor=[5, 7], degree=3)

    # Connect patches
    mesh.connect()

    # Define boundary conditions
    mesh.set_reflective_conditions(("left", "bottom", "right"))

    # Finalize mesh
    mesh.finalize()

    # Run checks
    run_eig(mesh, xs_server, 1.0, 256)


def test_square_fixed_source():
    # Get XS data
    total = 1  # 1/cm
    scattering_ratio = 0.9
    xs_server = Server(
        {
            "Material": {
                "total": np.array([total]),
                "scatter_gtg": np.array([[[total * scattering_ratio]]]),
            }
        }
    )

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
    patch = Patch(cad.bilinear(points), "Material")

    # Add uniform source of 1/cm to patch
    source = IsotropicInternalSource(np.ones((1, *patch.shape)))
    patch.set_source(source)

    # Create mesh
    mesh = IGAMesh()
    mesh.add_patch(patch)

    # Refine mesh resolution
    mesh.refine(factor=13, degree=3)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    run_fixed_source(mesh, xs_server, 0.420974, 256)
