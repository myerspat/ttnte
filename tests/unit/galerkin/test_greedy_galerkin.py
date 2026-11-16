"""GreedyGalerkin (TransportSolve.GALERKIN): one transport solve by residual-greedy
Galerkin sampling of directions (Guo & Peng), against the exact sweep of the same state.

The moment error tracks the tolerance, the solve is exact once every direction
is swept (or the basis spans the space), runs are deterministic for a seed, p
and q set the sweep counts, both starts work, and whole fixed-source solves
match EXACT transport.
"""

import pytest
import torch

from ttnte.sampling import Weighting
from ttnte.galerkin import (
    GalerkinConfig,
    GalerkinPhiOld,
    GalerkinSample,
    GalerkinStart,
    SvdNorm,
)
from ttnte.sweep import InnerSolver, SweepConfig, TransportSolve

from ..sweep.problems import BUILDERS, driver, quadrature, quarter_circle

G = 3


def _sampler(build, num_dim, n, sweep=None, **galerkin):
    """A GALERKIN solver mid-solve (nonzero, unconverged sources and traces), its
    sampler, and the exact moments and fibers of the current state.

    The few
    iterations reaching that state run at a loose tolerance: at a tight one every
    transport solve would sweep (nearly) every direction.
    """
    mesh, xs = build()
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=1e-2,
        max_iter=3,
        verbose=False,
        transport=TransportSolve.GALERKIN,
        galerkin=GalerkinConfig(**galerkin),
        **(sweep or {}),
    )
    sw.setup(quadrature(num_dim, n), cfg)
    sw.solve_fixed_source()
    D = sw.ordinates.shape[0]
    fibers = sw.sweep_fibers([(list(range(D)), list(range(G)))]).reshape(D, G, -1)
    moments = torch.einsum("dm,d,dgl->mgl", sw.harmonics, sw.weights, fibers)
    return sw, sw.get_galerkin(0), moments, fibers


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _all(sw):
    D = sw.ordinates.shape[0]
    return list(range(D)), list(range(G)), list(range(sw.fiber_length))


@pytest.fixture(scope="module")
def circle():
    """The reflective quarter circle, 4 x 4 refined, 256 directions: low rank in angle
    relative to the directions and the space."""
    return _sampler(lambda: quarter_circle(G, True, factor=4), 2, 8)


def test_error_tracks_tolerance(circle):
    """The moment error stays below tol, and fewer directions are swept for a looser
    tol."""
    sw, g, ref, _ = circle
    fibers = []
    for tol in (1e-2, 1e-3, 1e-4):
        g.set_tolerance(tol)
        g.run()
        assert g.converged
        assert _rel(g.moments(), ref) <= tol
        assert g.history[-1].residual <= tol and g.history[-1].phi_change <= tol
        fibers.append(g.num_fibers)
    D = sw.ordinates.shape[0]
    assert fibers[0] < fibers[1] < fibers[2] < D * G
    assert all(f % G == 0 for f in fibers)  # whole directions only


def test_moments_match_evaluate(circle):
    """Moments() is the quadrature of evaluate() over every direction."""
    sw, g, _, _ = circle
    g.set_tolerance(1e-3)
    g.run()
    psi = g.evaluate(*_all(sw))
    ref = torch.einsum("dm,d,dgl->mgl", sw.harmonics, sw.weights, psi)
    assert _rel(g.moments(), ref) < 1e-13


def test_swept_directions_are_exact(circle):
    """Swept directions keep their own fibers (their MGS coordinates, Remark 3.3)."""
    sw, g, _, fibers = circle
    g.set_tolerance(1e-2)
    g.run()
    swept = [d for d, s in enumerate(g.swept) if s]
    psi = g.evaluate(swept, list(range(G)), list(range(sw.fiber_length)))
    assert _rel(psi, fibers[swept]) < 1e-11


def test_exact_once_every_direction_is_swept():
    """1-D slab, 8 directions, tol at rounding: every direction is swept and the flux is
    the exact sweep."""
    sw, g, ref, fibers = _sampler(lambda: BUILDERS[1](G, True), 1, 2)
    g.set_tolerance(1e-14)
    g.run()
    assert all(g.swept) and g.converged and g.error == 0.0
    assert _rel(g.evaluate(*_all(sw)), fibers) < 1e-13
    assert _rel(g.moments(), ref) < 1e-13


def test_saturated_basis():
    """The octant sphere (81 unknowns per direction, 128 directions): once the basis
    spans the space, further snapshots add no vector (basis_rtol) and the reduced
    systems stay nonsingular; the flux is exact."""
    sw, g, ref, _ = _sampler(lambda: BUILDERS[3](G, True), 3, 4)
    g.set_tolerance(1e-13)
    g.run()
    assert g.ranks == [G * sw.fiber_length]
    assert g.dropped_snapshots > 0 and not all(g.swept)
    assert _rel(g.moments(), ref) < 1e-12


def test_seed_is_deterministic():
    build = lambda: quarter_circle(G, True)  # noqa: E731
    runs = []
    for seed in (5, 5, 6):
        sw, g, _, _ = _sampler(build, 2, 4, seed=seed)
        g.set_tolerance(1e-3)
        g.run()
        runs.append((g.moments(), list(g.swept)))
    assert torch.equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]
    assert runs[0][1] != runs[2][1]


@pytest.mark.parametrize("per_iter, candidates", [(1, 8), (2, 8), (3, 3)])
def test_per_iter_sets_sweep_counts(per_iter, candidates):
    """p directions start the run and are added per iteration: the swept fibers grow
    by p G per greedy iteration."""
    sw, g, _, _ = _sampler(
        lambda: quarter_circle(G, True, factor=4),
        2,
        8,
        per_iter=per_iter,
        candidates=candidates,
    )
    g.set_tolerance(1e-3)
    g.run()
    counts = [h.fibers for h in g.history]
    assert counts[0] == per_iter * G
    assert all(b - a == per_iter * G for a, b in zip(counts, counts[1:]))
    assert g.num_fibers == counts[-1]


def test_start_modes():
    """INITIAL_DIRECTIONS starts every run from the same directions (the cross's),
    RANDOM from p directions drawn by the seed."""
    build = lambda: quarter_circle(G, True, factor=4)  # noqa: E731
    starts = {}
    for start in (GalerkinStart.INITIAL_DIRECTIONS, GalerkinStart.RANDOM):
        for seed in (1, 2):
            sw, g, _, _ = _sampler(build, 2, 8, start=start, seed=seed, max_iter=1)
            g.set_tolerance(1e-3)
            g.run()
            assert len(g.history) == 1 and not g.converged
            starts[(start, seed)] = [d for d, s in enumerate(g.swept) if s]
    initial = starts[(GalerkinStart.INITIAL_DIRECTIONS, 1)]
    assert initial == starts[(GalerkinStart.INITIAL_DIRECTIONS, 2)]
    assert len(initial) > 1
    assert len(starts[(GalerkinStart.RANDOM, 1)]) == 1
    assert starts[(GalerkinStart.RANDOM, 1)] != starts[(GalerkinStart.RANDOM, 2)]


def test_max_iter_still_returns_a_flux():
    """A run capped by max_iter is unconverged but holds the flux of its basis."""
    sw, g, ref, _ = _sampler(
        lambda: quarter_circle(G, True, factor=4), 2, 8, max_iter=3
    )
    g.set_tolerance(1e-8)
    g.run()
    assert len(g.history) == 3 and not g.converged
    assert g.holds_flux and torch.isfinite(g.moments()).all()
    assert _rel(g.moments(), ref) < 1.0


def test_weighting_defaults_and_paper_setting():
    """All factors default on, like the cross; angle=False (Guo & Peng's equal direction
    weights) is a run option, and both converge to tol."""
    assert GalerkinConfig().weighting.angle
    sw, g, ref, _ = _sampler(
        lambda: quarter_circle(G, True, factor=4),
        2,
        8,
        weighting=Weighting(angle=False),
    )
    assert not g.weighting.angle and g.weighting.space and not g.weighting.energy
    g.set_tolerance(1e-3)
    g.run()
    assert g.converged and _rel(g.moments(), ref) <= 1e-3


@pytest.mark.parametrize(
    "inner, phi_old, expect",
    [
        (InnerSolver.GMRES, GalerkinPhiOld.AUTO, GalerkinPhiOld.WITHIN_RUN),
        (InnerSolver.RICHARDSON, GalerkinPhiOld.AUTO, GalerkinPhiOld.PREVIOUS_RUN),
        (InnerSolver.GMRES, GalerkinPhiOld.PREVIOUS_RUN, GalerkinPhiOld.PREVIOUS_RUN),
        (InnerSolver.RICHARDSON, GalerkinPhiOld.WITHIN_RUN, GalerkinPhiOld.WITHIN_RUN),
    ],
)
def test_phi_old_resolution(inner, phi_old, expect):
    """AUTO resolves at setup to the previous run's flux under source iteration (Guo &
    Peng: phi_old := phi^(n-1), this run's input) and to the run's own full solves under
    GMRES (Han et al.: each input is a Krylov vector); an explicit choice is kept."""
    mesh, xs = BUILDERS[1](G, False)
    sw = driver(1, mesh, xs)
    cfg = SweepConfig(
        verbose=False,
        inner_solver=inner,
        transport=TransportSolve.GALERKIN,
        galerkin=GalerkinConfig(phi_old=phi_old),
    )
    sw.setup(quadrature(1), cfg)
    assert sw.get_galerkin(0).config.phi_old == expect


def _fills(g):
    """The records of g's last run that formed the full flux ((3.7) passed)."""
    return [h for h in g.history if h.residual <= g.tolerance]


@pytest.mark.parametrize(
    "phi_old", [GalerkinPhiOld.WITHIN_RUN, GalerkinPhiOld.PREVIOUS_RUN]
)
def test_phi_old_reference(phi_old):
    """A second run on the same state: under WITHIN_RUN its first full solve has
    nothing to compare with (the previous run's flux is ignored), so the run needs
    a second one; under PREVIOUS_RUN the first is compared with the previous
    run's flux. Both converge to tol."""
    sw, g, ref, _ = _sampler(
        lambda: quarter_circle(G, True, factor=4), 2, 8, phi_old=phi_old
    )
    g.set_tolerance(1e-3)
    for _ in range(2):
        g.run()
        assert g.converged and _rel(g.moments(), ref) <= 1e-3
        fills = _fills(g)
        assert fills[-1].phi_change <= 1e-3
    first = fills[0].phi_change
    if phi_old == GalerkinPhiOld.WITHIN_RUN:
        assert first != first  # NaN: no reference at the run's start
        assert len(fills) >= 2
    else:
        assert first == first  # compared with the first run's flux


@pytest.mark.parametrize("norm", [SvdNorm.FROBENIUS, SvdNorm.NUCLEAR])
def test_svd_truncation(norm):
    """Guo & Peng's truncated SVD after a run (svd_tol): every run records the truncated
    rank; with keep_angular_flux, compress() (which the solver calls after reading the
    moments and traces) makes the held flux the rank-r' one, to rounding at a tiny
    svd_tol and to about svd_tol otherwise (1e-3, 1e-2), the moments staying the
    untruncated fill's; without it compress() changes nothing.

    The truncation never reaches the iteration, so runs of one seed on one state
    are identical and the truncations are of the same coefficients.
    """
    tol = 1e-4

    def run(**kw):
        kw.setdefault("keep_angular_flux", True)
        kw.setdefault("svd_norm", norm)
        sw, g, _, _ = _sampler(lambda: quarter_circle(G, True, factor=2), 2, 8, **kw)
        g.set_tolerance(tol)
        g.run()
        g.compress()
        return sw, g

    sw, full = run()
    psi = full.evaluate(*_all(sw))
    assert full.truncated_rank == -1 and full.record.truncated_rank == -1
    r = full.ranks[0]

    _, exact = run(svd_tol=1e-15)
    assert exact.ranks == [r] and exact.truncated_rank <= r
    assert _rel(exact.evaluate(*_all(sw)), psi) < 1e-12
    assert _rel(exact.moments(), full.moments()) < 1e-13

    ranks = []
    for svd_tol in (1e-3, 1e-2):
        _, cut = run(svd_tol=svd_tol)
        k = cut.truncated_rank
        assert cut.record.truncated_rank == k and cut.ranks == [r]
        assert 0 < k < r
        # The held flux is rank k; the moments are the untruncated ones
        assert _rel(cut.evaluate(*_all(sw)), psi) < 10 * svd_tol * r**0.5
        assert _rel(cut.moments(), full.moments()) < 1e-13
        ranks.append(k)
    assert ranks[1] < ranks[0]

    # Without keep_angular_flux: the rank only, the held flux untruncated
    _, rank_only = run(svd_tol=1e-3, keep_angular_flux=False)
    assert rank_only.truncated_rank == ranks[0]
    assert _rel(rank_only.evaluate(*_all(sw)), psi) < 1e-13


def test_svd_truncation_leaves_the_solve_unchanged():
    """Inside the sweep solver (reflective: lagged traces are read from the held
    flux) a truncation with keep_angular_flux changes only the flux kept, not the
    iteration: the same scalar flux as without, while the records report the
    truncated ranks."""
    solves = []
    for svd_tol in (0.0, 1e-3):
        sw, g, _, _ = _sampler(
            lambda: quarter_circle(G, True, factor=2),
            2,
            8,
            keep_angular_flux=True,
            svd_tol=svd_tol,
        )
        phi = torch.cat([sw.get_scalar_flux(i).reshape(-1) for i in sw.local_gids])
        solves.append((phi, [r.truncated_rank for r in sw.sampling_records], sw))
    (phi0, ranks0, _), (phi1, ranks1, sw1) = solves
    assert _rel(phi1, phi0) < 1e-13
    assert set(ranks0) == {-1} and all(0 < k for k in ranks1)
    assert all(k <= max(r.ranks) for k, r in zip(ranks1, sw1.sampling_records)) and any(
        k < max(r.ranks) for k, r in zip(ranks1, sw1.sampling_records)
    )


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(per_iter=0), "per_iter"),
        (dict(per_iter=4, candidates=2), "candidates"),
        (dict(mgs_max_projections=1), "mgs_max_projections"),
        (dict(mgs_eps=0.0), "mgs_eps"),
        (dict(basis_rtol=-1.0), "basis_rtol"),
        (dict(max_iter=-1), "max_iter"),
        (dict(svd_tol=-1.0), "svd_tol"),
        (dict(svd_tol=1.0), "svd_tol"),
    ],
)
def test_rejects_bad_config(kw, match):
    mesh, xs = BUILDERS[1](G, False)
    sw = driver(1, mesh, xs)
    cfg = SweepConfig(
        verbose=False, transport=TransportSolve.GALERKIN, galerkin=GalerkinConfig(**kw)
    )
    with pytest.raises(RuntimeError, match=match):
        sw.setup(quadrature(1), cfg)


# =================================================================
# Whole solves


def _phi(num_dim, transport, n=4, **kw):
    mesh, xs = BUILDERS[num_dim](G, True)
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(tol=1e-6, max_iter=500, verbose=False, transport=transport, **kw)
    sw.setup(quadrature(num_dim, n), cfg)
    sw.solve_fixed_source()
    return torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids]), sw


def test_fixed_source_richardson_matches_exact():
    """Source iteration with GALERKIN transport solves (phi_old carried across solves,
    the paper's setting) converges to the EXACT solution."""
    kw = dict(inner_solver=InnerSolver.RICHARDSON)
    exact, _ = _phi(1, TransportSolve.EXACT, **kw)
    phi, sw = _phi(1, TransportSolve.GALERKIN, **kw)
    assert sw.converged
    assert _rel(phi, exact) < 1e-4


@pytest.mark.parametrize("num_dim", [1, 2, 3])
@pytest.mark.parametrize("set_size", [0, 2])
def test_fixed_source_matches_exact(num_dim, set_size):
    """GMRES with GALERKIN transport solves (relaxed tolerances, as for the cross)
    converges to the EXACT solution, with one or two group sets (tolerances as in the
    cross's regression tests)."""
    exact, _ = _phi(num_dim, TransportSolve.EXACT, group_set_size=set_size)
    phi, sw = _phi(num_dim, TransportSolve.GALERKIN, group_set_size=set_size)
    assert sw.converged
    assert _rel(phi, exact) < 1e-4
    assert len(sw.sampling_records) > 0


@pytest.mark.parametrize("inner", [InnerSolver.GMRES, InnerSolver.RICHARDSON])
@pytest.mark.parametrize("num_dim", [2, 3])
def test_carry_basis_matches_exact(num_dim, inner):
    """carry_basis (our ablation, not the paper): every run after the first starts from
    the previous run's basis, rotated and cut at its own tolerance.

    The solve still converges to the EXACT solution, the records report the
    carried ranks (none for the first run, never more than the previous run's
    rank), and the runs sweep no more fibers in total than without carrying.
    """
    kw = dict(inner_solver=inner)
    exact, _ = _phi(num_dim, TransportSolve.EXACT, **kw)
    plain, sw0 = _phi(num_dim, TransportSolve.GALERKIN, **kw)
    phi, sw = _phi(
        num_dim,
        TransportSolve.GALERKIN,
        galerkin=GalerkinConfig(carry_basis=True),
        **kw,
    )
    assert sw.converged
    assert _rel(phi, exact) < 1e-4
    records = sw.sampling_records
    carried = [r.carried_rank for r in records]
    assert carried[0] == 0 and all(k > 0 for k in carried[1:])
    assert all(k <= max(prev.ranks) for k, prev in zip(carried[1:], records[:-1]))
    assert all(r.carried_rank == 0 for r in sw0.sampling_records)
    fibers = lambda s: sum(r.fibers for r in s.sampling_records)  # noqa: E731
    assert fibers(sw) <= fibers(sw0)


@pytest.mark.parametrize("relax", [True, False])
def test_richardson_relaxes_sampled_tolerance(relax):
    """Source iteration relaxes the sampled solves like GMRES does (sampling_relax):

    each sweep asks for sampling_inner_factor times the last change of the
    moments, sampling_max_tol at the first and never below tol_min =
    sampling_inner_factor times the inner tolerance; without relaxation every
    solve gets tol_min. Either way the solve converges to the EXACT solution.
    """
    kw = dict(inner_solver=InnerSolver.RICHARDSON, sampling_relax=relax)
    exact, _ = _phi(2, TransportSolve.EXACT, **kw)
    phi, sw = _phi(2, TransportSolve.GALERKIN, **kw)
    assert sw.converged and _rel(phi, exact) < 1e-4
    tols = [r.tolerance for r in sw.sampling_records]
    tol_min = sw.config.sampling_inner_factor * 0.1 * sw.config.tol
    if relax:
        assert tols[0] == pytest.approx(sw.config.sampling_max_tol)
        assert min(tols) >= tol_min * (1 - 1e-12) and min(tols) < tols[0]
    else:
        assert all(t == pytest.approx(tol_min) for t in tols)


def test_svd_norms():
    """At the same svd_tol the nuclear-norm cut (Guo & Peng's sum of singular values)
    keeps at least as many vectors as the Frobenius one (the cross's rule, the default),
    on the same coefficients.

    Only the sampler is kept: it must keep its
    driver alive (keep_alive in get_galerkin).
    """
    assert GalerkinConfig().svd_norm == SvdNorm.FROBENIUS
    ranks = {}
    for norm in (SvdNorm.FROBENIUS, SvdNorm.NUCLEAR):
        _, g, _, _ = _sampler(
            lambda: quarter_circle(G, True, factor=2), 2, 8, svd_tol=1e-3, svd_norm=norm
        )
        g.set_tolerance(1e-4)
        g.run()
        ranks[norm] = g.truncated_rank
    assert 0 < ranks[SvdNorm.FROBENIUS] <= ranks[SvdNorm.NUCLEAR]


# =================================================================
# Pair sampling (GalerkinSample.PAIR, our ablation)

PAIR = dict(sample=GalerkinSample.PAIR)


@pytest.fixture(scope="module")
def circle_pairs():
    return _sampler(lambda: quarter_circle(G, True, factor=4), 2, 8, **PAIR)


def test_pairs_error_tracks_tolerance(circle_pairs):
    """Pair sampling: the moment error stays below tol, fewer pairs are swept for a
    looser tol, and a sample is one fiber (any count, not whole directions)."""
    sw, g, ref, _ = circle_pairs
    fibers = []
    for tol in (1e-2, 1e-3, 1e-4):
        g.set_tolerance(tol)
        g.run()
        assert g.converged
        assert _rel(g.moments(), ref) <= tol
        fibers.append(g.num_fibers)
    assert fibers[0] < fibers[1] < fibers[2] < sw.ordinates.shape[0] * G
    assert len(g.swept) == sw.ordinates.shape[0] * G


def test_pairs_swept_are_exact_and_moments_match(circle_pairs):
    """A swept pair keeps its own fiber; moments() is the quadrature of evaluate()."""
    sw, g, _, fibers = circle_pairs
    g.set_tolerance(1e-3)
    g.run()
    swept = [u for u, s in enumerate(g.swept) if s]
    assert swept
    space = list(range(sw.fiber_length))
    for u in swept:
        d, k = divmod(u, G)
        assert _rel(g.evaluate([d], [k], space)[0, 0], fibers[d, k]) < 1e-11
    psi = g.evaluate(*_all(sw))
    ref = torch.einsum("dm,d,dgl->mgl", sw.harmonics, sw.weights, psi)
    assert _rel(g.moments(), ref) < 1e-13


def test_pairs_exact_once_every_pair_is_swept():
    """1-D slab, tol at rounding: every pair is swept and the flux is the exact
    sweep."""
    sw, g, ref, fibers = _sampler(lambda: BUILDERS[1](G, True), 1, 2, **PAIR)
    g.set_tolerance(1e-14)
    g.run()
    assert all(g.swept) and g.converged and g.error == 0.0
    assert _rel(g.evaluate(*_all(sw)), fibers) < 1e-13
    assert _rel(g.moments(), ref) < 1e-13


@pytest.mark.parametrize("inner", [InnerSolver.GMRES, InnerSolver.RICHARDSON])
@pytest.mark.parametrize("set_size", [0, 2])
def test_pairs_fixed_source_matches_exact(inner, set_size):
    """Whole multigroup solves with pair sampling converge to the EXACT solution."""
    kw = dict(inner_solver=inner, group_set_size=set_size)
    exact, _ = _phi(2, TransportSolve.EXACT, **kw)
    phi, sw = _phi(2, TransportSolve.GALERKIN, galerkin=GalerkinConfig(**PAIR), **kw)
    assert sw.converged
    assert _rel(phi, exact) < 1e-4


def test_one_group_pairs_equal_directions():
    """With one group a pair is a direction: the same solve, sample by sample."""
    out = {}
    for sample in (GalerkinSample.DIRECTION, GalerkinSample.PAIR):
        mesh, xs = BUILDERS[2](1, True)
        sw = driver(2, mesh, xs)
        cfg = SweepConfig(
            tol=1e-6,
            max_iter=500,
            verbose=False,
            transport=TransportSolve.GALERKIN,
            galerkin=GalerkinConfig(sample=sample),
        )
        sw.setup(quadrature(2, 4), cfg)
        sw.solve_fixed_source()
        phi = torch.cat([sw.get_scalar_flux(i).flatten() for i in sw.local_gids])
        out[sample] = (phi, [r.fibers for r in sw.sampling_records])
    (phi_d, fib_d), (phi_p, fib_p) = out.values()
    assert fib_d == fib_p
    assert _rel(phi_p, phi_d) < 1e-12
