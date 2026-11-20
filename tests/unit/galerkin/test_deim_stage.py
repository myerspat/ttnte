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

from ..sweep.problems import BUILDERS, driver, quadrature, quarter_circle, rel

G = 3


# EXACT references, solved once per problem and settings (read only by the tests)
_EXACT = {}


def _solve(num_dim, n, transport, eigenvalue=False, **kw):
    if transport == TransportSolve.EXACT:
        key = (num_dim, n, eigenvalue, tuple(sorted(kw.items())))
        if key not in _EXACT:
            _EXACT[key] = _run(num_dim, n, transport, eigenvalue, **kw)
        return _EXACT[key]
    return _run(num_dim, n, transport, eigenvalue, **kw)


def _run(num_dim, n, transport, eigenvalue=False, **kw):
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


def _check_records(sw, set_size, pairs=False):
    """The records tell the stages apart: greedy runs first, then DEIM runs that sweep
    exactly their DEIM directions (fewer than all), settled only after an update within
    the solver's stop tolerance (Han et al.'s eps_phi) of the previous outer iterate
    (for the first update, the greedy iterate of the switch); the last run of the
    solve is settled. With pairs a DEIM unit is one (direction, group) fiber. A DEIM
    run reports the finite Lebesgue constant of the selection it interpolated with
    (greedy runs none), so with one group set it changes only after an update."""
    D = sw.ordinates.shape[0] * (G if pairs else 1)
    recs = sw.sampling_records
    stages = [r.stage for r in recs]
    assert GalerkinStage.DEIM_UPDATE in stages
    first = stages.index(next(s for s in stages if s != GalerkinStage.GREEDY))
    assert all(s == GalerkinStage.GREEDY for s in stages[:first])
    sets = (
        [G] if not set_size else [min(set_size, G - k) for k in range(0, G, set_size)]
    )
    previous = None
    for r in recs:
        if r.stage == GalerkinStage.GREEDY:
            assert r.settled and r.deim_directions == 0
            assert math.isnan(r.lebesgue)
            continue
        assert math.isfinite(r.lebesgue)
        if not set_size and previous is not None:
            if previous.stage == GalerkinStage.DEIM_INNER:
                assert r.lebesgue == previous.lebesgue
        previous = r
        assert 1 <= r.deim_directions <= D
        if pairs:
            assert r.fibers == r.deim_directions
        else:
            assert any(r.fibers == r.deim_directions * g for g in sets)
        assert r.lebesgue >= 1.0 - 1e-12
        if r.stage == GalerkinStage.DEIM_INNER:
            assert not r.settled and math.isnan(r.outer_change)
            continue
        assert not math.isnan(r.outer_change)
        assert r.stop_tolerance > 0.0
        assert r.settled == (r.outer_change <= r.stop_tolerance)
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
    assert rel(phi, exact) < 1e-5
    _check_records(sw, set_size)


@pytest.mark.parametrize("num_dim, n", [(1, 16), (2, 6)])
def test_paper_matches_exact(num_dim, n):
    """Han et al.'s version (the defaults: switch_tol 1e-4, inner_tol 1e-6), one group
    set."""
    exact, _ = _solve(num_dim, n, TransportSolve.EXACT)
    phi, sw = _solve(num_dim, n, TransportSolve.GALERKIN, galerkin=_deim())
    assert sw.converged
    assert rel(phi, exact) < 1e-5
    _check_records(sw, 0)


def test_paper_diverges_in_3d():
    """Without DSA the paper's interpolated source iteration diverges in 3-D.

    Han et al. have DSA inside the inner iterations; without it, on the 3-D
    octant at switch_tol 1e-4 (|I| = 81 of 128, Lebesgue constants 7-27), the
    flux grows until the divergence test reports it (DEIM_DIVERGENCE_GROWTH
    times its norm at the switch).
    """
    with pytest.raises(RuntimeError, match="DEIM source iteration diverged"):
        _solve(3, 4, TransportSolve.GALERKIN, galerkin=_deim())


def test_paper_later_switch_in_3d():
    """Switching later (switch_tol 1e-7) the same problem converges."""
    exact, _ = _solve(3, 4, TransportSolve.EXACT)
    phi, sw = _solve(3, 4, TransportSolve.GALERKIN, galerkin=_deim(switch_tol=1e-7))
    assert sw.converged and rel(phi, exact) < 1e-5
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
        assert sw.converged and rel(phi, exact) < 1e-5


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
    assert sw.converged and rel(phi, exact) < 1e-5
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
    assert sw.converged and rel(phi, exact) < 1e-5
    _check_records(sw, 0)


@pytest.mark.parametrize("deim", [{}, OURS], ids=["paper", "ours"])
def test_eigenvalue_matches_exact(deim):
    exact, ex = _solve(1, 16, TransportSolve.EXACT, eigenvalue=True)
    phi, sw = _solve(
        1, 16, TransportSolve.GALERKIN, eigenvalue=True, galerkin=_deim(**deim)
    )
    assert sw.converged
    assert abs(sw.k_eff - ex.k_eff) < 1e-6
    assert rel(phi / phi.norm(), exact / exact.norm()) < 1e-5
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
            _deim(switch_tol=0.0),
            "switch_tol",
        ),
    ],
    ids=["gmres", "switch-tol"],
)
def test_rejects_unsupported(kw, galerkin, match):
    mesh, xs = BUILDERS[1](G, True)
    sw = driver(1, mesh, xs)
    cfg = SweepConfig(
        verbose=False, transport=TransportSolve.GALERKIN, galerkin=galerkin, **kw
    )
    with pytest.raises(RuntimeError, match=match):
        sw.setup(quadrature(1, 4), cfg)


# ----------------------------------------------------------------------
# DEIM over (direction, group) pairs (sample = PAIR, ours, a labelled ablation):
# V has a row per pair, DEIM picks pairs, the angular update couples the groups
# through the transfer blocks on the spatial basis.


def _pair_deim(**kw):
    return GalerkinConfig(
        sample=GalerkinSample.PAIR, deim=DeimConfig(enabled=True, **kw)
    )


@pytest.mark.parametrize("num_dim, n, set_size", CASES)
def test_pairs_ours_matches_exact(num_dim, n, set_size):
    """With our additions DEIM over pairs converges to EXACT in 1/2/3-D, one and two
    group sets, sweeping far fewer pairs than directions x groups."""
    kw = dict(group_set_size=set_size)
    exact, _ = _solve(num_dim, n, TransportSolve.EXACT, **kw)
    phi, sw = _solve(
        num_dim, n, TransportSolve.GALERKIN, galerkin=_pair_deim(**OURS), **kw
    )
    assert sw.converged
    assert rel(phi, exact) < 1e-5
    recs = _check_records(sw, set_size, pairs=True)
    units = sw.ordinates.shape[0] * (set_size or G)
    assert max(r.deim_directions for r in recs) < units // 2


@pytest.mark.parametrize(
    "num_dim, n, set_size, switch_tol",
    [(1, 16, 0, 1e-4), (1, 16, 2, 1e-4), (2, 6, 0, 1e-7), (3, 4, 0, 1e-7)],
)
def test_pairs_paper_matches_exact(num_dim, n, set_size, switch_tol):
    """The paper's settings over pairs converge in 1-D (one and two sets) and, as over
    directions, in 2-D and 3-D with a late switch (no DSA here)."""
    kw = dict(group_set_size=set_size)
    exact, _ = _solve(num_dim, n, TransportSolve.EXACT, **kw)
    phi, sw = _solve(
        num_dim,
        n,
        TransportSolve.GALERKIN,
        galerkin=_pair_deim(switch_tol=switch_tol),
        **kw,
    )
    assert sw.converged
    assert rel(phi, exact) < 1e-5
    _check_records(sw, set_size, pairs=True)


@pytest.mark.parametrize("deim", [{}, OURS], ids=["paper", "ours"])
def test_one_group_pairs_equal_directions(deim):
    """With one group a pair is a direction: the same DEIM solve, run by run (eq.
    3.5's pair weights w_d e_0^2 only scale V)."""
    out = {}
    for sample in (GalerkinSample.DIRECTION, GalerkinSample.PAIR):
        mesh, xs = BUILDERS[2](1, True)
        sw = driver(2, mesh, xs)
        cfg = SweepConfig(
            tol=1e-6,
            max_iter=500,
            verbose=False,
            transport=TransportSolve.GALERKIN,
            inner_solver=InnerSolver.RICHARDSON,
            galerkin=GalerkinConfig(
                sample=sample, deim=DeimConfig(enabled=True, **deim)
            ),
        )
        sw.setup(quadrature(2, 4), cfg)
        sw.solve_fixed_source()
        phi = torch.cat([sw.get_scalar_flux(i).flatten() for i in sw.local_gids])
        recs = sw.sampling_records
        out[sample] = (phi, [(r.stage, r.fibers, r.deim_directions) for r in recs])
    (phi_d, rec_d), (phi_p, rec_p) = out.values()
    assert rec_d == rec_p
    assert any(stage == GalerkinStage.DEIM_UPDATE for stage, _, _ in rec_d)
    assert rel(phi_p, phi_d) < 1e-10


def test_pairs_truncated_spatial_basis():
    """On the step-3 meshes the pair greedy spans the whole spatial space (|I| = N_s);
    on a finer quarter circle (N_s = 400) the DEIM pairs stay fewer than N_s, so the
    stage interpolates on a truncated spatial basis, and still matches EXACT to the
    accuracy of the switch (ours, switch 1e-4)."""
    out = {}
    for transport, galerkin in [
        (TransportSolve.EXACT, None),
        (TransportSolve.GALERKIN, _pair_deim(**OURS)),
    ]:
        mesh, xs = quarter_circle(G, True, factor=8)
        sw = driver(2, mesh, xs)
        cfg = SweepConfig(
            tol=1e-6,
            max_iter=500,
            verbose=False,
            transport=transport,
            inner_solver=InnerSolver.RICHARDSON,
            **({} if galerkin is None else dict(galerkin=galerkin)),
        )
        sw.setup(quadrature(2, 6), cfg)
        sw.solve_fixed_source()
        out[transport] = (
            torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids]),
            sw,
        )
    (exact, _), (phi, sw) = out.values()
    assert sw.converged
    assert rel(phi, exact) < 1e-5
    recs = _check_records(sw, 0, pairs=True)
    assert 0 < max(r.deim_directions for r in recs) < sw.fiber_length


def test_zero_flux_stays_greedy(monkeypatch):
    """A set whose flux is zero (here: no source at all) never switches to DEIM.

    It has nothing to interpolate, and its zero run change is no sign of a
    settled iteration (it used to enter the stage on rank-0 coefficients and
    fail).
    """
    from ..sweep import problems

    monkeypatch.setattr(
        problems,
        "_source",
        lambda G_: problems.FixedSource(
            isotropic_strength=torch.zeros(G_, dtype=torch.float64)
        ),
    )
    phi, sw = _solve(
        1, 16, TransportSolve.GALERKIN, galerkin=_deim(), sampling_relax=False
    )
    assert sw.converged and phi.abs().max() == 0.0
    assert all(r.stage == GalerkinStage.GREEDY for r in sw.sampling_records)


def test_thread_count_does_not_change_the_solve():
    """One thread or four, the hybrid takes the same runs (stages and fibers: no thread-
    dependent selection; the reduced assembly runs on one MKL thread) and the same flux
    to rounding (the sampler's own products use its threads, and even EXACT differs in
    the last bits, about 1e-15, between thread counts)."""
    out = []
    for threads in (1, 4):
        phi, sw = _solve(
            2,
            6,
            TransportSolve.GALERKIN,
            galerkin=_deim(**OURS),
            num_threads=threads,
            sampling_threads=threads,
        )
        out.append((phi, [(r.stage, r.fibers) for r in sw.sampling_records]))
    assert out[0][1] == out[1][1]
    assert rel(out[0][0], out[1][0]) < 1e-13
