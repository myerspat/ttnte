"""Multi-rank sweeps reproduce the single-rank sweeper.

Every rank also solves the whole problem alone (``serial=True``, on
MPI_COMM_SELF) as the reference, then the distributed scalar flux is gathered
by GID and compared.

Run with ``mpirun -n {2,3,4} pytest --with-mpi
tests/regression/multi_patch/test_sweep_mpi.py``.
"""

import mpi4py.MPI
import pytest
import torch
from igakit import cad

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BCPlane, BoundaryType
from ttnte.sweep import (
    GroupIteration,
    IGASweepDriver2D,
    InnerSolver,
    LocalSolverType,
    SweepConfig,
)
from ttnte.xs.benchmarks import kaist

from .test_sweep import _qset, _quarter_circle, _xs

CPU = torch.device("cpu")
COMM = mpi4py.MPI.COMM_WORLD


def _lattice(fills, factor, split=False):
    """Quarter circle of fuel surrounded by water (no fixed source), for
    k-eigenvalue runs: reflective on x = 0 and y = 0, vacuum outside."""
    c0 = cad.circle(radius=5, angle=torch.pi / 2)
    c1, c2 = c0.slice(0, 0, 0.5), c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    l1 = cad.line(p0=(6, 0), p1=(6, 6))
    l2 = cad.line(p0=(6, 6), p1=(0, 6))
    surfaces = [
        cad.ruled(l0, c1),
        cad.ruled(l0, c2),
        cad.ruled(c1, l1),
        cad.ruled(c2, l2),
    ]
    mesh = IGAMesh(mpi_context)
    for k, surface in enumerate(surfaces):
        patch = Patch.from_igakit(
            cad.refine(surface, factor, 2),
            device=CPU,
            dtype=torch.float64,
            fill=fills[2] if k < 2 else fills[8],
        )
        for block in patch.split_elements() if split else [patch]:
            mesh.add_block(block)
    mesh.connect()
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True), BoundaryType.REFLECTIVE, tol=1e-6
    )
    mesh.finalize()
    return mesh


def _solve(mesh, xs, qset, serial, eigen=False, partition=None, **kw):
    """Solve and return ({gid: scalar flux} of this rank's patches, k, driver)."""
    sweeper = IGASweepDriver2D(mesh, xs, mpi_context, serial=serial)
    if partition is not None:
        sweeper.set_partition(partition)
    sweeper.setup(qset, SweepConfig(tol=1e-11, max_iter=5000, verbose=False, **kw))
    if eigen:
        sweeper.solve_eigenvalue(1e-12)
    else:
        sweeper.solve_fixed_source()
    assert sweeper.converged
    gids = [b.gid for b in sweeper.mesh.blocks]
    phi = {g: sweeper.get_scalar_flux(g).clone() for g in gids}
    return phi, sweeper.k_eff, sweeper


def _gather(phi):
    merged = {}
    for part in COMM.allgather({g: p.numpy() for g, p in phi.items()}):
        assert not set(part) & set(merged), "a GID is owned by two ranks"
        merged.update(part)
    return {g: torch.from_numpy(p) for g, p in merged.items()}


def _rel_diff(a, b):
    assert set(a) == set(b)
    num = sum(((a[g] - b[g]) ** 2).sum().item() for g in b)
    return (num / sum((b[g] ** 2).sum().item() for g in b)) ** 0.5


def _round_robin(mesh):
    """GID -> rank cycling over the ranks: maximal cross-rank interfaces."""
    return {b.gid: b.gid % COMM.size for b in mesh.blocks}


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(directions_per_task=1, num_threads=4),
        dict(
            group_iteration=GroupIteration.JACOBI,
            inner_solver=InnerSolver.RICHARDSON,
            inner_max_iter=1,
        ),
        dict(store_angular_flux=True, cache_factorizations=True, num_threads=2),
    ],
)
def test_fixed_source_pdg(kw):
    """Curved PDG quarter circle, METIS partition."""
    torch.set_default_dtype(torch.float64)
    fills, xs = _xs()
    qset = _qset(4)
    ref, _, _ = _solve(_quarter_circle(fills, 3, False), xs, qset, True, **kw)
    phi, _, sweeper = _solve(_quarter_circle(fills, 3, False), xs, qset, False, **kw)
    assert len(sweeper.mesh.blocks) < 4  # actually distributed
    assert _rel_diff(_gather(phi), ref) < 1e-10


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("num_threads", [1, 4])
def test_fixed_source_fdg_cyclic_cut(num_threads):
    """FDG elements of the curved geometry partitioned round-robin, so interfaces whose
    upwind graph is cyclic (lagged edges) cross ranks."""
    torch.set_default_dtype(torch.float64)
    fills, xs = _xs()
    qset = _qset(4)
    kw = dict(local_solver=LocalSolverType.DENSE_LU, num_threads=num_threads)
    ref, _, serial = _solve(_quarter_circle(fills, 2, True), xs, qset, True, **kw)
    assert serial.graph.num_weight_lagged + serial.graph.num_cycle_lagged > 0
    mesh = _quarter_circle(fills, 2, True)
    phi, _, _ = _solve(mesh, xs, qset, False, partition=_round_robin(mesh), **kw)
    assert _rel_diff(_gather(phi), ref) < 1e-10


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(group_set_size=1, num_threads=4, directions_per_task=2),
        dict(group_set_size=3, group_iteration=GroupIteration.JACOBI),
    ],
)
def test_eigenvalue(kw):
    """7-group k-eigenvalue (fuel + water with upscatter), FDG round-robin."""
    torch.set_default_dtype(torch.float64)
    fills, xs = kaist(problem="2B", device=CPU, dtype=torch.float64)
    qset = _qset(2)
    ref, k_ref, _ = _solve(_lattice(fills, 1, True), xs, qset, True, eigen=True, **kw)
    mesh = _lattice(fills, 1, True)
    phi, k, _ = _solve(
        mesh, xs, qset, False, eigen=True, partition=_round_robin(mesh), **kw
    )
    assert abs(k - k_ref) < 1e-10
    assert _rel_diff(_gather(phi), ref) < 1e-9
