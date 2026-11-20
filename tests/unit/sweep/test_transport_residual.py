"""The transport residual source + inflow - L psi of one direction, assembled by the
sweep kernel's own helpers: a fiber the oracle sweeps has a residual at rounding level
(on curved, Cartesian, reflective, cycle-broken and non-conforming meshes, with every
direct local solver), and the residual is affine in psi."""

import pytest
import torch

from ttnte.sweep import LocalSolverType, SweepConfig

from . import test_nonconforming as nc
from .problems import BUILDERS, boxes, driver, mid_solve, quadrature, quarter_circle

G = 3


def _relative_residuals(sw, groups):
    """Per direction, ||r(fiber)|| / ||r(0)||: the swept fiber's residual relative to
    the problem's source and lagged inflow."""
    nd = sw.ordinates.shape[0]
    fibers = sw.sweep_fibers([(list(range(nd)), groups)]).reshape(nd, len(groups), -1)
    zero = torch.zeros_like(fibers[0])
    out = []
    for d in range(nd):
        r = sw.transport_residual(d, groups, fibers[d])
        r0 = sw.transport_residual(d, groups, zero)
        out.append((r.norm() / r0.norm()).item())
    return out


CASES = [(d, r) for d in (1, 2, 3) for r in (False, True)]


@pytest.mark.parametrize("num_dim, reflective", CASES)
def test_swept_fibers_have_no_residual(num_dim, reflective):
    """Curved meshes (quarter circle, octant sphere split into elements), vacuum and
    reflective."""
    mesh, xs = BUILDERS[num_dim](G, reflective)
    sw = mid_solve(mesh, xs, num_dim)
    assert max(_relative_residuals(sw, list(range(G)))) < 1e-12


@pytest.mark.parametrize("reflective", [False, True])
def test_cycle_broken_by_lagged_edges(reflective):
    """The quarter circle split into its elements, with 4 x 4 directions: its upwind
    graph has cycles, broken by lagged internal edges whose inflow is the stored trace,
    not the candidate flux."""
    mesh, xs = quarter_circle(G, reflective, split=True)
    sw = driver(2, mesh, xs)
    sw.setup(quadrature(2, 4), SweepConfig(tol=1e-14, max_iter=3, verbose=False))
    sw.solve_fixed_source()
    assert sw.graph.num_weight_lagged + sw.graph.num_cycle_lagged > 0
    assert sw.num_delayed_blocks > 0
    assert max(_relative_residuals(sw, list(range(G)))) < 1e-12


@pytest.mark.parametrize("num_dim", [2, 3])
@pytest.mark.parametrize("reflective", [False, True])
def test_cartesian_kronecker(num_dim, reflective):
    """Axis-aligned boxes, solved by the Kronecker-sum solver (SYLVESTER) the residual
    does not use: the pair-list operator is the same one."""
    mesh, xs = boxes(num_dim, G, reflective)
    sw = mid_solve(mesh, xs, num_dim)
    assert sw.num_kronecker_units > 0
    assert max(_relative_residuals(sw, list(range(G)))) < 1e-12


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize(
    "solver",
    [LocalSolverType.BANDED_LU, LocalSolverType.DENSE_LU, LocalSolverType.SYLVESTER],
)
def test_local_solvers(solver, cache):
    """With cache_factorizations the fibers are solved from factors cached by the
    earlier sweeps (no operator assembled), the residual always assembles L: the two
    must agree."""
    mesh, xs = BUILDERS[2](G, True)
    sw = mid_solve(mesh, xs, 2, local_solver=solver, cache_factorizations=cache)
    if cache and solver != LocalSolverType.SYLVESTER:
        assert sw.factorization_bytes > 0
    assert max(_relative_residuals(sw, list(range(G)))) < 1e-12


@pytest.mark.parametrize(
    "surfaces, num_dim",
    [
        (lambda: nc.squares(False), 2),
        (lambda: nc.rings(False), 2),
        (lambda: nc.rings(False, flip=True), 2),
        (lambda: nc.cylinder_rings(False), 3),
    ],
    ids=["squares", "rings", "rings-flipped", "cylinder-rings"],
)
def test_nonconforming_interfaces(surfaces, num_dim):
    """Inflow across a non-conforming interface goes through its coupling pairs."""
    fills, xs = nc.absorber()
    mesh = nc.build(surfaces(), fills)
    sw = mid_solve(mesh, xs, num_dim)
    assert max(_relative_residuals(sw, list(range(nc.G)))) < 1e-12


def test_group_subset_and_order():
    """Any groups, in any order: each group's residual is its own."""
    mesh, xs = BUILDERS[2](G, True)
    sw = mid_solve(mesh, xs, 2)
    assert max(_relative_residuals(sw, [2, 0])) < 1e-12
    nd = sw.ordinates.shape[0]
    psi = torch.randn(G, sw.fiber_length, generator=torch.Generator().manual_seed(0))
    d = nd // 3
    whole = sw.transport_residual(d, [0, 1, 2], psi)
    part = sw.transport_residual(d, [2, 0], psi[[2, 0]])
    assert torch.allclose(part, whole[[2, 0]], rtol=0, atol=1e-13 * whole.abs().max())


def test_residual_is_affine_and_sees_perturbations():
    """r(x) = b - A x: r(x + y) + r(0) = r(x) + r(y), and perturbing one patch of a swept
    fiber gives a residual of the perturbation's size."""
    mesh, xs = BUILDERS[2](G, False)
    sw = mid_solve(mesh, xs, 2)
    groups = list(range(G))
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(G, sw.fiber_length, generator=gen)
    y = torch.randn(G, sw.fiber_length, generator=gen)
    d = 1
    r = lambda v: sw.transport_residual(d, groups, v)
    lhs, rhs = r(x + y) + r(torch.zeros_like(x)), r(x) + r(y)
    assert ((lhs - rhs).norm() / rhs.norm()).item() < 1e-13

    fiber = sw.sweep_fibers([([d], groups)])
    off = sw.fiber_offsets
    bumped = fiber.clone()
    bumped[:, off[1] : off[2]] *= 1.01
    rel = (r(bumped).norm() / r(torch.zeros_like(fiber)).norm()).item()
    assert rel > 1e-4


def test_rejects_bad_arguments():
    mesh, xs = BUILDERS[1](G, False)
    sw = mid_solve(mesh, xs, 1)
    nd = sw.ordinates.shape[0]
    psi = torch.zeros(2, sw.fiber_length)
    with pytest.raises(RuntimeError, match="Direction out of range"):
        sw.transport_residual(nd, [0, 1], psi)
    with pytest.raises(RuntimeError, match="distinct and in range"):
        sw.transport_residual(0, [1, 1], psi)
    with pytest.raises(RuntimeError, match="distinct and in range"):
        sw.transport_residual(0, [0, G], psi)
    with pytest.raises(ValueError, match="fiber_length"):
        sw.transport_residual(0, [0], psi)
