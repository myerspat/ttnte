"""GreedyGalerkin with GalerkinConfig.deim (Han et al. 2026, step H3): greedy source
iterations until the scalar flux settles to deim.switch_tol, then DEIM source iterations
(the DEIM directions swept, the rest interpolated) with angular updates (every direction
solved with its within-set scattering on X = orth(Psi_I), giving V and the next
directions; the iterate stays the interpolant, as in the paper).

The paper's version converges to the EXACT solution with one group set in 1-D
and 2-D (fixed source and eigenvalue) and diverges in 3-D unless it switches
later (no DSA here; theirs has it). Our labelled additions, galerkin_flux (the
update's run holds X L), update_on_growth (a growing inner change updates at
once) and inner_tol = 0 (the run's tolerance), each change what they claim, and
together converge everywhere (1-D/2-D/3-D, one and two group sets, DEIM and
Q-DEIM). Disabling the stage changes nothing, and the unsupported settings are
rejected.
"""

import math

import pytest
import torch

from ttnte.galerkin import (
    DeimConfig,
    DeimSelect,
    GalerkinConfig,
    GalerkinSample,
    GalerkinStage,
)
from ttnte.sweep import InnerSolver, SweepConfig, TransportSolve

from ..sweep.problems import BUILDERS, driver, quadrature

G = 3


def _solve(num_dim, n, transport, eigenvalue=False, **kw):
    mesh, xs = BUILDERS[num_dim](G, True, fissile=eigenvalue)
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=1e-6,
        max_iter=500,
        verbose=False,
        transport=transport,
        inner_solver=InnerSolver.RICHARDSON,
        **kw,
    )
    sw.setup(quadrature(num_dim, n), cfg)
    if eigenvalue:
        sw.solve_eigenvalue(1e-8)
    else:
        sw.solve_fixed_source()
    return torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids]), sw


def _deim(select=DeimSelect.DEIM, **kw):
    """The paper's DEIM stage (our additions off unless asked for)."""
    return GalerkinConfig(deim=DeimConfig(enabled=True, select=select, **kw))


# Ours (labelled): the update's flux is X L, a growing inner change updates at once,
# and the inner test is the run's (relaxed) tolerance: with X L each update is a
# reduced scattering solve, so early, frequent updates pay; with the paper's
# interpolant the relaxed inner test lets a drifting stage loosen itself
OURS = dict(galerkin_flux=True, update_on_growth=True, inner_tol=0.0)


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _check_records(sw, set_size):
    """The records tell the stages apart: greedy runs first, then DEIM runs that sweep
    exactly their DEIM directions (fewer than all), settled only after an update within
    the run's tolerance of the previous one; the last run of the solve is settled."""
    D = sw.ordinates.shape[0]
    recs = sw.sampling_records
    stages = [r.stage for r in recs]
    assert GalerkinStage.DEIM_UPDATE in stages
    first = stages.index(next(s for s in stages if s != GalerkinStage.GREEDY))
    assert all(s == GalerkinStage.GREEDY for s in stages[:first])
    sets = (
        [G] if not set_size else [min(set_size, G - k) for k in range(0, G, set_size)]
    )
    for r in recs:
        if r.stage == GalerkinStage.GREEDY:
            assert r.settled and r.deim_directions == 0
            continue
        assert 1 <= r.deim_directions <= D
        assert any(r.fibers == r.deim_directions * g for g in sets)
        assert r.lebesgue >= 1.0 - 1e-12
        if r.stage == GalerkinStage.DEIM_INNER:
            assert not r.settled and math.isnan(r.outer_change)
        elif r.settled:
            assert r.outer_change <= r.tolerance
    assert recs[-1].settled
    return recs


CASES = [(1, 16, 0), (1, 16, 2), (2, 6, 0), (2, 6, 2), (3, 4, 0)]


@pytest.mark.parametrize("num_dim, n, set_size", CASES)
def test_ours_matches_exact(num_dim, n, set_size):
    """With both of our additions the hybrid converges to EXACT.

    The DEIM stage keeps the greedy rank of the switch, so its accuracy is set
    there (runs at most switch_tol = 1e-4 accurate): a few 1e-6 here, against
    the solver tolerance 1e-6.
    """
    kw = dict(group_set_size=set_size)
    exact, _ = _solve(num_dim, n, TransportSolve.EXACT, **kw)
    phi, sw = _solve(num_dim, n, TransportSolve.GALERKIN, galerkin=_deim(**OURS), **kw)
    assert sw.converged
    assert _rel(phi, exact) < 1e-5
    _check_records(sw, set_size)


@pytest.mark.parametrize("num_dim, n", [(1, 16), (2, 6)])
def test_paper_matches_exact(num_dim, n):
    """Han et al.'s version (the defaults: switch_tol 1e-4, inner_tol 1e-6), one group
    set."""
    exact, _ = _solve(num_dim, n, TransportSolve.EXACT)
    phi, sw = _solve(num_dim, n, TransportSolve.GALERKIN, galerkin=_deim())
    assert sw.converged
    assert _rel(phi, exact) < 1e-5
    _check_records(sw, 0)


def test_paper_diverges_in_3d():
    """Without DSA (Han et al.

    have it inside the inner iterations) the paper's interpolated source
    iteration diverges on the 3-D octant at switch_tol 1e-4 (|I| = 81 of 128,
    Lebesgue constants 7-27): the flux grows until its norm overflows, reported
    as such.
    """
    with pytest.raises(RuntimeError, match="DEIM source iteration diverged"):
        _solve(3, 4, TransportSolve.GALERKIN, galerkin=_deim())


def test_paper_later_switch_in_3d():
    """Switching later (switch_tol 1e-7) the same problem converges."""
    exact, _ = _solve(3, 4, TransportSolve.EXACT)
    phi, sw = _solve(3, 4, TransportSolve.GALERKIN, galerkin=_deim(switch_tol=1e-7))
    assert sw.converged and _rel(phi, exact) < 1e-5
    _check_records(sw, 0)


def test_galerkin_flux_changes_the_iterate():
    """deim.galerkin_flux (ours) leaves everything up to the first angular update alone
    (same records) and changes the flux from there; both converge."""
    exact, _ = _solve(1, 16, TransportSolve.EXACT)
    phi0, sw0 = _solve(1, 16, TransportSolve.GALERKIN, galerkin=_deim())
    phi1, sw1 = _solve(
        1, 16, TransportSolve.GALERKIN, galerkin=_deim(galerkin_flux=True)
    )
    r0, r1 = sw0.sampling_records, sw1.sampling_records
    first = [r.stage for r in r0].index(GalerkinStage.DEIM_UPDATE)
    for a, b in zip(r0[: first + 1], r1[: first + 1]):
        assert (a.stage, a.fibers) == (b.stage, b.fibers)
        assert a.run_change == b.run_change or (
            math.isnan(a.run_change) and math.isnan(b.run_change)
        )
    assert not torch.equal(phi0, phi1)
    for phi, sw in [(phi0, sw0), (phi1, sw1)]:
        assert sw.converged and _rel(phi, exact) < 1e-5


def test_update_on_growth():
    """1-D, two group sets, inner_tol = 0 (ours: the run's relaxed tolerance): the
    thermal set's interpolated source iteration grows (by 2.4% a run, |I| = 13, Lebesgue
    constant 4.8), never reaches its inner tolerance, so never updates, and diverges.

    With deim.update_on_growth (ours) every grown run is an update, and the
    solve converges.
    """
    kw = dict(group_set_size=2)
    with pytest.raises(RuntimeError, match="DEIM source iteration diverged"):
        _solve(1, 16, TransportSolve.GALERKIN, galerkin=_deim(inner_tol=0.0), **kw)
    exact, _ = _solve(1, 16, TransportSolve.EXACT, **kw)
    phi, sw = _solve(
        1,
        16,
        TransportSolve.GALERKIN,
        galerkin=_deim(inner_tol=0.0, update_on_growth=True),
        **kw,
    )
    assert sw.converged and _rel(phi, exact) < 1e-5
    grown = [r for r in sw.sampling_records if r.inner_diverged]
    assert grown and all(r.stage == GalerkinStage.DEIM_UPDATE for r in grown)


def test_interpolates_with_fewer_directions():
    """In 1-D (32 directions) the DEIM directions are a proper subset: the flux of the
    others is interpolated, and the DEIM runs sweep fewer fibers than the greedy
    ones."""
    _, sw = _solve(1, 16, TransportSolve.GALERKIN, galerkin=_deim())
    recs = sw.sampling_records
    deim = [r for r in recs if r.stage != GalerkinStage.GREEDY]
    assert max(r.deim_directions for r in deim) < sw.ordinates.shape[0]
    g = sw.get_galerkin(0)
    assert g.in_deim_stage and g.num_updates > 0
    assert len(g.deim_directions) == len(set(g.deim_directions))


def test_qdeim_matches_exact():
    exact, _ = _solve(2, 6, TransportSolve.EXACT)
    phi, sw = _solve(
        2, 6, TransportSolve.GALERKIN, galerkin=_deim(DeimSelect.QDEIM, **OURS)
    )
    assert sw.converged and _rel(phi, exact) < 1e-5
    _check_records(sw, 0)


@pytest.mark.parametrize("deim", [{}, OURS], ids=["paper", "ours"])
def test_eigenvalue_matches_exact(deim):
    exact, ex = _solve(1, 16, TransportSolve.EXACT, eigenvalue=True)
    phi, sw = _solve(
        1, 16, TransportSolve.GALERKIN, eigenvalue=True, galerkin=_deim(**deim)
    )
    assert sw.converged
    assert abs(sw.k_eff - ex.k_eff) < 1e-6
    assert _rel(phi / phi.norm(), exact / exact.norm()) < 1e-5
    _check_records(sw, 0)


def test_explicit_inner_tol():
    """A loose deim.inner_tol updates the angle after fewer inner iterations: more
    updates per inner run than with the run's tolerance (OURS)."""
    _, sw0 = _solve(1, 16, TransportSolve.GALERKIN, galerkin=_deim(**OURS))
    _, sw1 = _solve(
        1, 16, TransportSolve.GALERKIN, galerkin=_deim(**(OURS | dict(inner_tol=1e-3)))
    )
    assert sw1.converged

    def ratio(sw):
        stages = [r.stage for r in sw.sampling_records]
        return stages.count(GalerkinStage.DEIM_UPDATE) / max(
            1, stages.count(GalerkinStage.DEIM_INNER)
        )

    assert ratio(sw1) > ratio(sw0)


def test_disabled_changes_nothing():
    """deim.enabled = False (whatever its other fields) gives the same bits as the
    default configuration."""
    phi0, sw0 = _solve(2, 4, TransportSolve.GALERKIN, galerkin=GalerkinConfig())
    off = GalerkinConfig(deim=DeimConfig(enabled=False, switch_tol=0.5, **OURS))
    phi1, sw1 = _solve(2, 4, TransportSolve.GALERKIN, galerkin=off)
    assert torch.equal(phi0, phi1)
    assert [r.fibers for r in sw0.sampling_records] == [
        r.fibers for r in sw1.sampling_records
    ]
    assert all(r.stage == GalerkinStage.GREEDY for r in sw1.sampling_records)


@pytest.mark.parametrize(
    "kw, galerkin, match",
    [
        (dict(inner_solver=InnerSolver.GMRES), _deim(), "RICHARDSON"),
        (
            dict(inner_solver=InnerSolver.RICHARDSON),
            GalerkinConfig(sample=GalerkinSample.PAIR, deim=DeimConfig(enabled=True)),
            "direction sampling",
        ),
        (
            dict(inner_solver=InnerSolver.RICHARDSON),
            _deim(switch_tol=0.0),
            "switch_tol",
        ),
    ],
    ids=["gmres", "pairs", "switch-tol"],
)
def test_rejects_unsupported(kw, galerkin, match):
    mesh, xs = BUILDERS[1](G, True)
    sw = driver(1, mesh, xs)
    cfg = SweepConfig(
        verbose=False, transport=TransportSolve.GALERKIN, galerkin=galerkin, **kw
    )
    with pytest.raises(RuntimeError, match=match):
        sw.setup(quadrature(1, 4), cfg)
