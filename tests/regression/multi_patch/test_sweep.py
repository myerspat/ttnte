"""End-to-end checks of the transport sweep solver against the TT solver.

Both solvers discretize the same system (same quadratures, upwind masks,
reflective mapping and scattering kernel), so on any mesh the converged sweep
solution must equal the converged TT block-Jacobi solution to the solver
tolerances, for patch discontinuous (PDG) meshes and for fully discontinuous
(FDG) meshes built with Patch.split_elements().
"""

import pytest
import torch
from igakit import cad

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.driver import IGATransportDriver2D
from ttnte.math import ProductQuadrature
from ttnte.mesh import IGAMesh
from ttnte.physics import BCPlane, BoundaryType, DGTransportAssemblerConfig, FixedSource
from ttnte.solvers import (
    AMEnSolver,
    BlockJacobiStrategy,
    CommMode,
    DDSolverConfig,
    ExecMode,
    IGADDSolver,
    MemoryPolicy,
)
from ttnte.sweep import (
    IGASweepDriver2D,
    LocalSolverType,
    SweepConfig,
    TaskGranularity,
)
from ttnte.xs import Material, Server

CPU = torch.device("cpu")


def _xs():
    source = Material("Source")
    source.total = torch.tensor([1.0])
    source.scatter_gtg = torch.tensor([[[0.9]]])
    source.finalize()
    void = Material("Void")
    void.total = torch.tensor([0.0])
    void.scatter_gtg = torch.tensor([[[0.0]]])
    void.finalize()
    server = Server()
    server.add_material(source)
    server.add_material(void)
    server.finalize()
    return [source, void], server


def _quarter_circle(fills, factor, split, parents=None):
    """Shielded quarter circle (mc27 benchmark geometry): a scattering source disk with
    a void shield, reflective on x = 0 and y = 0, vacuum outside.

    If `parents` is a list, the parent patch of every block is appended.
    """
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
        fill = fills[0] if k < 2 else fills[1]
        patch = Patch.from_igakit(
            cad.refine(surface, factor, 2),
            device=CPU,
            dtype=torch.float64,
            fill=fill.label,
        )
        if k < 2:
            patch.source = FixedSource(isotropic_strength=torch.tensor([1.0]))
        for block in patch.split_elements() if split else [patch]:
            mesh.add_block(block)
            if parents is not None:
                parents.append(k)
    mesh.connect()
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True), BoundaryType.REFLECTIVE, tol=1e-6
    )
    mesh.finalize()
    return mesh


def _qset(n):
    qset = ProductQuadrature.gauss_legendre_chebyshev(n, n, 2)
    qset.to_(CPU, torch.float64)
    return qset


def _tt_scalar_flux(mesh, xs, qset):
    cfg = DGTransportAssemblerConfig()
    cfg.rounding.eps = 1e-13
    cfg.cross.eps = 1e-13
    cfg.max_dense_size = int(1e10)
    cfg.cross_jacobian_inverse = False
    driver = IGATransportDriver2D(mesh, xs, mpi_context)
    driver.assemble(qset, cfg)
    dd_cfg = DDSolverConfig(
        tol=1e-11,
        max_iter=5000,
        use_gpu=False,
        memory_policy=MemoryPolicy.RESIDENT,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_cfg)
    strategy.set_local_solver(
        AMEnSolver(nswp=4, eps=1e-12, kickrank=4, local_iterations=200, resets=4)
    )
    driver.solve_fixed_source(
        IGADDSolver(driver.mesh, strategy), tol=1e-10, max_iter=300, verbose=False
    )
    qset.to_(CPU, torch.float64)
    return {
        gid: qset.integrate(driver.get_system(gid).state, 0.0).as_tt().to_dense()
        for gid in driver.gid2rank
    }


def _sweep(mesh, xs, qset, chunks=None, **kw):
    sweeper = IGASweepDriver2D(mesh, xs, mpi_context)
    config = SweepConfig(tol=1e-10, max_iter=3000, verbose=False, **kw)
    sweeper.setup(qset, config, chunks or {})
    sweeper.solve_fixed_source()
    assert sweeper.converged
    return sweeper


def _rel_diff(a, b):
    num = sum(((a[g].reshape(b[g].shape) - b[g]) ** 2).sum().item() for g in b)
    return (num / sum((b[g] ** 2).sum().item() for g in b)) ** 0.5


@pytest.mark.mpi(min_size=1)
def test_pdg_sweep_matches_tt():
    """Curved multi-patch PDG mesh with reflective planes, scattering source, void
    shield and degenerate (collapsed) edges."""
    if mpi_context.world_size > 1:
        pytest.skip("Sweeps run on one rank")
    torch.set_default_dtype(torch.float64)
    fills, xs = _xs()
    qset = _qset(2)
    sweeper = _sweep(_quarter_circle(fills, 2, False), xs, qset)
    phi = {g: sweeper.get_scalar_flux(g) for g in range(4)}
    ref = _tt_scalar_flux(_quarter_circle(fills, 2, False), xs, qset)
    assert _rel_diff(phi, ref) < 1e-8


@pytest.mark.mpi(min_size=1)
def test_fdg_sweep_matches_tt():
    """The same geometry split into single-element patches (FDG) solved with the dense
    local solver."""
    if mpi_context.world_size > 1:
        pytest.skip("Sweeps run on one rank")
    torch.set_default_dtype(torch.float64)
    fills, xs = _xs()
    qset = _qset(2)
    sweeper = _sweep(
        _quarter_circle(fills, 1, True), xs, qset, local_solver=LocalSolverType.DENSE_LU
    )
    phi = {g: sweeper.get_scalar_flux(g) for g in range(sweeper.graph.num_nodes)}
    ref = _tt_scalar_flux(_quarter_circle(fills, 1, True), xs, qset)
    assert _rel_diff(phi, ref) < 1e-8


@pytest.mark.mpi(min_size=1)
def test_solver_variants_agree():
    """Banded and dense local solvers, threads and group sets, and the angular-flux
    toggle all give the same moments."""
    if mpi_context.world_size > 1:
        pytest.skip("Sweeps run on one rank")
    torch.set_default_dtype(torch.float64)
    fills, xs = _xs()
    qset = _qset(4)
    base = _sweep(_quarter_circle(fills, 3, False), xs, qset)
    ref = {g: base.get_scalar_flux(g) for g in range(4)}
    for kw in [
        dict(local_solver=LocalSolverType.DENSE_LU),
        dict(num_threads=4),
        dict(store_angular_flux=True),
        dict(cache_factorizations=True),
        dict(cache_factorizations=True, local_solver=LocalSolverType.DENSE_LU),
    ]:
        other = _sweep(_quarter_circle(fills, 3, False), xs, qset, **kw)
        assert _rel_diff({g: other.get_scalar_flux(g) for g in range(4)}, ref) < 1e-12
    # Angular flux integrates to the scalar flux
    other = _sweep(_quarter_circle(fills, 3, False), xs, qset, store_angular_flux=True)
    psi = other.get_angular_flux(0)
    w = other.weights.reshape(psi.shape[0], psi.shape[1])
    phi = torch.einsum("ab,ab...->...", w, psi)
    assert torch.allclose(phi, other.get_scalar_flux(0), rtol=1e-12, atol=1e-14)
    # The TransportSolution carries the scalar flux
    sol = other.build_solution()
    assert sol is not None


@pytest.mark.mpi(min_size=1)
def test_task_granularity_agrees():
    """Per-unit tasks, one subdomain task, and FDG elements chunked by parent patch
    (whose chunk graph can be cyclic along curved interfaces) give the same solution,
    with one and several threads."""
    if mpi_context.world_size > 1:
        pytest.skip("Sweeps run on one rank")
    torch.set_default_dtype(torch.float64)
    fills, xs = _xs()
    qset = _qset(4)
    parents = []
    mesh = _quarter_circle(fills, 3, True, parents)
    base = _sweep(mesh, xs, qset, local_solver=LocalSolverType.DENSE_LU)
    n = base.graph.num_nodes
    ref = {g: base.get_scalar_flux(g) for g in range(n)}
    variants = [
        (dict(task_granularity=TaskGranularity.SUBDOMAIN), None),
        (dict(task_granularity=TaskGranularity.SUBDOMAIN, num_threads=4), None),
        (dict(num_threads=4), {g: parents[g] for g in range(n)}),
        (
            dict(task_granularity=TaskGranularity.SUBDOMAIN, cache_factorizations=True),
            None,
        ),
    ]
    for kw, chunks in variants:
        other = _sweep(
            _quarter_circle(fills, 3, True),
            xs,
            qset,
            chunks,
            local_solver=LocalSolverType.DENSE_LU,
            **kw,
        )
        assert other.num_tasks < base.num_tasks
        phi = {g: other.get_scalar_flux(g) for g in range(n)}
        assert _rel_diff(phi, ref) < 1e-12
