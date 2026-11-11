"""ExtrudedSolver: the exact partially separable local solver of extruded 3-D patches
(Schur along the extruded axis, one in-plane band solve per mode)."""

import math

import pytest
import torch
from igakit import cad

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.sweep import ExtrudedSolver, LocalSolverType, SweepConfig

from .problems import CPU, _finish, _source, boxes, driver, multigroup_xs, quadrature
from .problems import octant_sphere

G = 3


def annulus(r0=1.0, r1=3.0):
    inner = cad.circle(radius=r0, angle=math.pi / 2)
    outer = cad.circle(radius=r1, angle=math.pi / 2)
    return cad.ruled(inner, outer)


def extruded(height=2.0, factor=(3, 2, 3)):
    """Quarter annulus extruded along +z: parametric axis 2 -> physical z."""
    return cad.refine(cad.extrude(annulus(), height, 2), factor=factor, degree=2)


def tapered():
    """Outer radius 3 at z = 0, 2 at z = 2: not extruded."""
    top = annulus(1.0, 2.0)
    top.move(2.0, 2)
    return cad.refine(cad.ruled(annulus(), top), factor=(3, 2, 3), degree=2)


def mesh_of(volumes, reflective=False):
    fills, xs = multigroup_xs(G)
    mesh = IGAMesh(mpi_context)
    for k, vol in enumerate(volumes):
        patch = Patch.from_igakit(
            vol, device=CPU, dtype=torch.float64, fill=fills[0 if k == 0 else 1].label
        )
        if k == 0:
            patch.source = _source(G)
        mesh.add_block(patch)
    return _finish(mesh, 3, reflective), xs


def discretization(vol):
    mesh, xs = mesh_of([vol])
    sw = driver(3, mesh, xs)
    sw.setup(quadrature(3, 1), SweepConfig(verbose=False))
    return sw.get_discretization(sw.local_gids[0])


# The same extruded patch with its extruded axis at each parametric position, and
# rotated so it runs along physical x
GEOMETRIES = {
    "z, parametric axis 2": (lambda: extruded(), 2, 2),
    "z, parametric axis 0": (lambda: extruded().transpose([2, 0, 1]), 0, 2),
    "x, parametric axis 2": (lambda: extruded().rotate(math.pi / 2, 1), 2, 0),
}

DIRECTIONS = [
    [0.48, 0.6, 0.64],
    [-0.6, 0.48, -0.64],
    [0.8, -0.59, 0.1],
    [-0.3, -0.5, 0.81],
]


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_detects_the_extruded_axis(name):
    build, axis, physical = GEOMETRIES[name]
    solver = ExtrudedSolver.try_create(discretization(build()))
    assert solver is not None
    assert (solver.axis, solver.physical_axis) == (axis, physical)


@pytest.mark.parametrize("build", [tapered, lambda: None])
def test_rejects_patches_that_are_not_extruded(build):
    if build() is None:
        mesh, xs = octant_sphere(G, split=False)
        sw = driver(3, mesh, xs)
        sw.setup(quadrature(3, 1), SweepConfig(verbose=False))
        disc = sw.get_discretization(sw.local_gids[0])
    else:
        disc = discretization(build())
    assert ExtrudedSolver.try_create(disc) is None


@pytest.mark.parametrize("name", list(GEOMETRIES))
@pytest.mark.parametrize("sigma", [0.05, 3.0])
@pytest.mark.parametrize("path", ["diagonal", "schur"])
def test_solve_matches_the_assembled_operator(name, sigma, path):
    build, _, physical = GEOMETRIES[name]
    disc = discretization(build())
    # max_eigvec_cond = 0 forces the Schur path
    solver = ExtrudedSolver.try_create(
        disc, max_eigvec_cond=1e8 if path == "diagonal" else 0.0
    )
    directions = [list(d) for d in DIRECTIONS]
    # No flow along the extruded axis (the decoupled, real path)
    flat = [0.0, 0.0, 0.0]
    flat[(physical + 1) % 3], flat[(physical + 2) % 3] = 0.6, -0.8
    directions.append(flat)
    torch.manual_seed(0)
    b = torch.randn(disc.num_dofs, dtype=torch.float64)
    for d in directions:
        omega = torch.tensor(d, dtype=torch.float64)
        omega = omega / omega.norm()
        L = disc.local_operator(omega, sigma)
        x_ref = torch.linalg.solve(L, b)
        x = solver.solve(omega, sigma, b)
        assert ((x - x_ref).norm() / x_ref.norm()).item() < 1e-10


def _solved(volumes, solver, reflective=False):
    mesh, xs = mesh_of(volumes, reflective)
    sw = driver(3, mesh, xs)
    cfg = SweepConfig(tol=1e-11, max_iter=500, verbose=False, local_solver=solver)
    sw.setup(quadrature(3, 2), cfg)
    sw.solve_fixed_source()
    assert sw.converged
    return sw


def _stacked_downward():
    """Two levels extruded downward from their tops (as the Big Ten builder), fuel below
    moderator, different axial refinements."""
    out = []
    for (z0, z1), nz in [((0.0, 3.0), 3), ((3.0, 5.0), 2)]:
        vol = cad.extrude(annulus().translate(z1, 2), -(z1 - z0), 2)
        out.append(cad.refine(vol, factor=(3, 2, nz), degree=2))
    return out


@pytest.mark.parametrize("reflective", [False, True])
def test_sylvester_sweep_matches_banded_lu_on_extruded_patches(reflective):
    lu = _solved(_stacked_downward(), LocalSolverType.BANDED_LU, reflective)
    sy = _solved(_stacked_downward(), LocalSolverType.SYLVESTER, reflective)
    assert sy.num_extruded_units == len(sy.local_gids)
    assert sy.num_kronecker_units == 0
    a = torch.cat([sy.get_moments(g).flatten() for g in sy.local_gids])
    b = torch.cat([lu.get_moments(g).flatten() for g in lu.local_gids])
    assert ((a - b).norm() / b.norm()).item() < 1e-10


def test_affine_boxes_keep_the_kronecker_solver():
    mesh, xs = boxes(3, G)
    sw = driver(3, mesh, xs)
    sw.setup(
        quadrature(3, 1),
        SweepConfig(verbose=False, local_solver=LocalSolverType.SYLVESTER),
    )
    assert sw.num_kronecker_units == len(sw.local_gids)
    assert sw.num_extruded_units == 0
