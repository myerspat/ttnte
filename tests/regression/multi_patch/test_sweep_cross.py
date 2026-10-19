"""TransportSolve.CROSS against the exact sweeper: the low-rank transport solves must
reach the same discrete solution to the requested tolerance, in 1/2/3-D, with vacuum and
reflective boundaries, for fixed-source and k-eigenvalue problems."""

import pytest
import torch

from tests.unit.sweep.problems import BUILDERS, driver, quadrature
from ttnte.cross import CrossConfig
from ttnte.sweep import InnerSolver, SweepConfig, TransportSolve

G = 6
TOL = 1e-6


def _solve(num_dim, reflective, transport, fissile=False, n=3, **kw):
    cross = kw.pop("cross", CrossConfig())
    mesh, xs = BUILDERS[num_dim](G, reflective, fissile)
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=TOL,
        max_iter=300,
        verbose=False,
        num_threads=4,
        transport=transport,
        cross=cross,
        **kw,
    )
    sw.setup(quadrature(num_dim, n), cfg)
    if fissile:
        sw.solve_eigenvalue(1e-8)
    else:
        sw.solve_fixed_source()
    assert sw.converged
    return sw


def _scalar_flux(sw):
    return torch.cat([sw.get_moments(gid)[0].flatten() for gid in sw.local_gids])


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


CASES = [(d, r) for d in (1, 2, 3) for r in (False, True)]


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective", CASES)
def test_fixed_source_matches_exact(num_dim, reflective):
    exact = _solve(num_dim, reflective, TransportSolve.EXACT)
    cross = _solve(num_dim, reflective, TransportSolve.CROSS)
    assert _rel(_scalar_flux(cross), _scalar_flux(exact)) < 10 * TOL
    assert len(cross.cross_fibers) > 0
    assert max(cross.cross_tolerances) <= CrossConfig().max_tol


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective", [(1, True), (2, True), (3, False)])
def test_eigenvalue_matches_exact(num_dim, reflective):
    exact = _solve(num_dim, reflective, TransportSolve.EXACT, fissile=True)
    cross = _solve(num_dim, reflective, TransportSolve.CROSS, fissile=True)
    assert abs(cross.k_eff - exact.k_eff) < 1e-5
    assert _rel(_scalar_flux(cross), _scalar_flux(exact)) < 1e-4


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective", [(1, True), (2, True), (3, False)])
@pytest.mark.parametrize("transport", [TransportSolve.EXACT, TransportSolve.CROSS])
def test_correction_form_fixed_source(num_dim, reflective, transport):
    exact = _solve(num_dim, reflective, TransportSolve.EXACT)
    corr = _solve(num_dim, reflective, transport, correction_form=True)
    assert _rel(_scalar_flux(corr), _scalar_flux(exact)) < 10 * TOL


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim, reflective", [(1, True), (2, True), (3, False)])
@pytest.mark.parametrize("transport", [TransportSolve.EXACT, TransportSolve.CROSS])
def test_correction_form_eigenvalue(num_dim, reflective, transport):
    exact = _solve(num_dim, reflective, TransportSolve.EXACT, fissile=True)
    corr = _solve(num_dim, reflective, transport, fissile=True, correction_form=True)
    assert abs(corr.k_eff - exact.k_eff) < 1e-5
    assert _rel(_scalar_flux(corr), _scalar_flux(exact)) < 1e-4
    # One transport solve per group set per outer (no right-hand side and
    # final sweeps): fewer group-set sweeps than the standard form
    if transport == TransportSolve.EXACT:
        assert corr.equivalent_sweeps < exact.equivalent_sweeps


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("transport", [TransportSolve.EXACT, TransportSolve.CROSS])
def test_correction_form_with_forcing(transport):
    # Loose forced inner tolerances skip GMRES in some outers; those must
    # still take the new fission source (accept y0) and not stall
    standard = _solve(2, True, TransportSolve.EXACT, fissile=True, inner_forcing=0.1)
    corr = _solve(
        2, True, transport, fissile=True, inner_forcing=0.1, correction_form=True
    )
    assert abs(corr.k_eff - standard.k_eff) < 1e-5
    assert _rel(_scalar_flux(corr), _scalar_flux(standard)) < 1e-4
    assert corr.num_iterations <= standard.num_iterations + 2


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize(
    "layout",
    [
        "polar,azimuthal,space|energy",
        "polar,azimuthal|space|energy",
        "energy|space|azimuthal,polar",
    ],
)
def test_layouts_and_richardson(layout):
    exact = _solve(2, True, TransportSolve.EXACT)
    cross = _solve(
        2,
        True,
        TransportSolve.CROSS,
        cross=CrossConfig(layout=layout),
        inner_solver=InnerSolver.RICHARDSON,
    )
    assert _rel(_scalar_flux(cross), _scalar_flux(exact)) < 10 * TOL


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("num_dim", [2, 3])
def test_patch_tts_match_angular_flux(num_dim):
    """keep_tt: each patch's TT in a split, reordered layout reproduces the
    exact angular flux of that patch."""
    layout = {
        2: "polar|azimuthal|y|x|energy",
        3: "energy|x,y,z|polar,azimuthal",
    }[num_dim]
    exact = _solve(num_dim, False, TransportSolve.EXACT, store_angular_flux=True)
    cross = _solve(
        num_dim,
        False,
        TransportSolve.CROSS,
        # Energy first: the angle bond grows from the initial directions by
        # one-site residual growth, which needs more sweeps
        cross=CrossConfig(layout=layout, keep_tt=True, max_sweeps=40),
    )
    order = layout.replace("|", ",").split(",")
    for gid in cross.local_gids:
        cores = cross.get_patch_angular_flux_tt(gid)
        t = cores[0]
        for c in cores[1:]:
            t = torch.tensordot(t, c, ([t.dim() - 1], [0]))
        # exact: (polar, azimuthal, x, y[, z], energy)
        ref = exact.get_angular_flux(gid)
        names = ["polar", "azimuthal", "x", "y", "z"][: ref.dim() - 1] + ["energy"]
        if num_dim == 2:
            names = ["polar", "azimuthal", "x", "y", "energy"]
        ref = ref.permute([names.index(a) for a in order])
        assert _rel(t.reshape(ref.shape), ref) < 1e-4


@pytest.mark.mpi(min_size=1)
def test_store_angular_flux_and_release():
    exact = _solve(2, True, TransportSolve.EXACT, store_angular_flux=True)
    cross = _solve(2, True, TransportSolve.CROSS, store_angular_flux=True)
    for gid in exact.local_gids:
        assert _rel(cross.get_angular_flux(gid), exact.get_angular_flux(gid)) < 1e-4
    # keep_tt is off: the TT is released after the solve
    with pytest.raises(RuntimeError):
        cross.get_patch_angular_flux_tt(cross.local_gids[0])


@pytest.mark.mpi(min_size=1)
def test_angular_flux_from_kept_tt_and_histories():
    exact = _solve(2, True, TransportSolve.EXACT, store_angular_flux=True)
    cross = _solve(2, True, TransportSolve.CROSS, cross=CrossConfig(keep_tt=True))
    for gid in exact.local_gids:
        assert _rel(cross.get_angular_flux(gid), exact.get_angular_flux(gid)) < 1e-4
    solution = cross.angular_solution()
    for gid in cross.local_gids:
        field = solution.get_local_field(gid).as_tt().to_dense()
        assert _rel(field.reshape(-1), cross.get_angular_flux(gid).reshape(-1)) < 1e-12
    assert len(cross.cross_ranks) == len(cross.cross_fibers)
    history = cross.cross_histories
    assert len(history) == len(cross.cross_fibers)
    assert all(len(run) >= 1 for run in history)
    first = history[0]
    assert first[-1]["ranks"] == cross.cross_ranks[0]
    assert [h["phase"] for h in first[:2]] == ["forward", "backward"]
    assert all(a["fibers"] <= b["fibers"] for a, b in zip(first, first[1:]))
