"""TransportSolve.GALERKIN against the exact sweeper: the greedy Galerkin transport
solves (Guo & Peng, under GMRES as in Han et al.) must reach the same discrete solution
to the requested tolerance, in 1/2/3-D, with vacuum and reflective boundaries, for
fixed-source and k-eigenvalue problems, under GMRES and source iteration, and record
every transport solve."""

import math

import mpi4py.MPI
import pytest
import torch

from tests.unit.sweep.problems import BUILDERS, driver, quadrature
from ttnte import mpi_context
from ttnte.galerkin import (
    DeimConfig,
    GalerkinConfig,
    GalerkinPhiOld,
    GalerkinRun,
    GalerkinStage,
)
from ttnte.sweep import InnerSolver, SweepConfig, TransportSolve

G = 6
TOL = 1e-6


def _config(transport, galerkin=None, **kw):
    return SweepConfig(
        tol=TOL,
        max_iter=300,
        verbose=False,
        num_threads=4,
        transport=transport,
        galerkin=galerkin or GalerkinConfig(),
        **kw,
    )


def _solve(num_dim, reflective, transport, fissile=False, n=3, **kw):
    # GALERKIN is serial in space (SweepSolver::build_samplers throws on more
    # than one rank, see test_rejects_multiple_ranks). Skip on every rank alike,
    # before any collective.
    if transport == TransportSolve.GALERKIN and mpi_context.world_size > 1:
        pytest.skip("GALERKIN transport solves are serial in space for now")
    mesh, xs = BUILDERS[num_dim](G, reflective, fissile)
    sw = driver(num_dim, mesh, xs)
    sw.setup(quadrature(num_dim, n), _config(transport, **kw))
    if fissile:
        sw.solve_eigenvalue(1e-8)
    else:
        sw.solve_fixed_source()
    assert sw.converged
    return sw


def _scalar_flux(sw):
    parts = [sw.get_moments(gid)[0].flatten() for gid in sw.local_gids]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _flux_rel(sw, ref):
    """Relative scalar-flux error over all ranks (a collective)."""
    a, b = _scalar_flux(sw), _scalar_flux(ref)
    sums = [((a - b) ** 2).sum().item(), (b**2).sum().item()]
    if mpi_context.world_size > 1:
        sums = [mpi4py.MPI.COMM_WORLD.allreduce(v) for v in sums]
    return math.sqrt(sums[0] / sums[1])


def _check_records(sw):
    """One GalerkinRun record per transport solve, consistent within itself."""
    records = sw.sampling_records
    assert len(records) > 0
    for rec in records:
        assert isinstance(rec, GalerkinRun)
        assert len(rec.history) > 0
        assert rec.history[-1].fibers <= rec.fibers
        assert rec.history[-1].rank <= rec.max_rank == rec.ranks[0]
        assert rec.dropped_snapshots >= 0
        assert rec.orthogonality_failures == 0
        assert rec.seconds >= rec.sweep_seconds >= 0.0
    assert max(r.tolerance for r in records) <= SweepConfig().sampling_max_tol
    assert sw.sampling_seconds >= sum(r.sweep_seconds for r in records)


CASES = [(d, r) for d in (1, 2, 3) for r in (False, True)]


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective", CASES)
def test_fixed_source_matches_exact(num_dim, reflective):
    exact = _solve(num_dim, reflective, TransportSolve.EXACT)
    galerkin = _solve(num_dim, reflective, TransportSolve.GALERKIN)
    assert _flux_rel(galerkin, exact) < 10 * TOL
    _check_records(galerkin)
    assert galerkin.get_galerkin(0).config.phi_old == GalerkinPhiOld.WITHIN_RUN


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective", [(1, True), (2, True), (3, False)])
def test_eigenvalue_matches_exact(num_dim, reflective):
    exact = _solve(num_dim, reflective, TransportSolve.EXACT, fissile=True)
    galerkin = _solve(num_dim, reflective, TransportSolve.GALERKIN, fissile=True)
    assert abs(galerkin.k_eff - exact.k_eff) < 1e-5
    assert _flux_rel(galerkin, exact) < 1e-4
    _check_records(galerkin)


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective", [(1, True), (2, True), (3, False)])
@pytest.mark.parametrize("fissile", [False, True])
def test_source_iteration(num_dim, reflective, fissile):
    """RICHARDSON (source iteration, Guo & Peng's setting): phi_old carried from the
    previous transport solve."""
    kw = dict(fissile=fissile, inner_solver=InnerSolver.RICHARDSON)
    exact = _solve(num_dim, reflective, TransportSolve.EXACT, **kw)
    galerkin = _solve(num_dim, reflective, TransportSolve.GALERKIN, **kw)
    assert galerkin.get_galerkin(0).config.phi_old == GalerkinPhiOld.PREVIOUS_RUN
    if fissile:
        assert abs(galerkin.k_eff - exact.k_eff) < 1e-5
        assert _flux_rel(galerkin, exact) < 1e-4
    else:
        assert _flux_rel(galerkin, exact) < 10 * TOL
    _check_records(galerkin)


# Han et al.'s DEIM stage as published, and with our labelled additions (the
# update's flux X L, an update on a growing inner change, the run's tolerance as the
# inner test). Without DSA the published
# version diverges on the 2-D/3-D problems, so it runs in 1-D only
DEIM_VARIANTS = [
    (1, True, "paper"),
    (1, True, "ours"),
    (2, True, "ours"),
    (3, False, "ours"),
]


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective, variant", DEIM_VARIANTS)
@pytest.mark.parametrize("fissile", [False, True])
def test_deim_source_iteration(num_dim, reflective, variant, fissile):
    """Han et al.'s hybrid (GalerkinConfig.deim): greedy source iterations, then DEIM
    ones with angular updates.

    The DEIM stage keeps the greedy rank of the switch, so its accuracy is set
    there (as in test_source_iteration's bounds).
    """
    kw = dict(fissile=fissile, inner_solver=InnerSolver.RICHARDSON)
    ours = variant == "ours"
    exact = _solve(num_dim, reflective, TransportSolve.EXACT, **kw)
    deim = _solve(
        num_dim,
        reflective,
        TransportSolve.GALERKIN,
        galerkin=GalerkinConfig(
            deim=DeimConfig(
                enabled=True,
                galerkin_flux=ours,
                update_on_growth=ours,
                inner_tol=0.0 if ours else 1e-6,
            )
        ),
        **kw,
    )
    if fissile:
        assert abs(deim.k_eff - exact.k_eff) < 1e-5
        assert _flux_rel(deim, exact) < 1e-4
    else:
        assert _flux_rel(deim, exact) < 10 * TOL
    records = deim.sampling_records
    stages = [r.stage for r in records]
    assert GalerkinStage.DEIM_UPDATE in stages and records[-1].settled
    for rec in records:
        assert isinstance(rec, GalerkinRun)
        if rec.stage == GalerkinStage.GREEDY:
            assert len(rec.history) > 0 and rec.deim_directions == 0
        else:
            assert rec.fibers == rec.deim_directions * G
            assert rec.lebesgue >= 1.0 - 1e-12


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective", [(1, True), (2, True), (3, False)])
@pytest.mark.parametrize("fissile", [False, True])
def test_correction_form(num_dim, reflective, fissile):
    exact = _solve(num_dim, reflective, TransportSolve.EXACT, fissile=fissile)
    corr = _solve(
        num_dim,
        reflective,
        TransportSolve.GALERKIN,
        fissile=fissile,
        correction_form=True,
    )
    if fissile:
        assert abs(corr.k_eff - exact.k_eff) < 1e-5
        assert _flux_rel(corr, exact) < 1e-4
    else:
        assert _flux_rel(corr, exact) < 10 * TOL


@pytest.mark.mpi(min_size=1)
def test_angular_flux_stored_kept_and_released():
    """store_angular_flux: psi from the last transport solve's low-rank flux;
    galerkin.keep_angular_flux: get_angular_flux evaluates it without stored psi;
    otherwise the flux is released after the solve."""
    exact = _solve(2, True, TransportSolve.EXACT, store_angular_flux=True)
    stored = _solve(2, True, TransportSolve.GALERKIN, store_angular_flux=True)
    kept = _solve(
        2,
        True,
        TransportSolve.GALERKIN,
        galerkin=GalerkinConfig(keep_angular_flux=True),
    )
    plain = _solve(2, True, TransportSolve.GALERKIN)
    for gid in exact.local_gids:
        ref = exact.get_angular_flux(gid)
        assert _rel(stored.get_angular_flux(gid), ref) < 1e-4
        assert _rel(kept.get_angular_flux(gid), ref) < 1e-4
    assert kept.get_galerkin(0).holds_flux
    assert not plain.get_galerkin(0).holds_flux
    with pytest.raises(RuntimeError, match="galerkin.keep_angular_flux"):
        plain.get_angular_flux(plain.local_gids[0])
    with pytest.raises(RuntimeError, match="No Galerkin sampler"):
        exact.get_galerkin()


@pytest.mark.mpi(min_size=2)
def test_rejects_multiple_ranks():
    """Setup throws on every rank alike (no rank is left inside a collective)."""
    mesh, xs = BUILDERS[2](G, True, False)
    sw = driver(2, mesh, xs)
    with pytest.raises(RuntimeError, match="serial in space"):
        sw.setup(quadrature(2, 3), _config(TransportSolve.GALERKIN))
    mpi4py.MPI.COMM_WORLD.Barrier()
