import pytest
import torch
import torchtt as tntt

from ttnte.linalg import (
    State,
    Operator,
    LinearSystem,
    TTEngine,
    Source,
    AMEnBackend,
    AMEnNativeOptions,
    AMEnPreconditioner,
    mv,
)
from ttnte.solvers import (
    AMEnSolver,
    StaticFreezePolicy,
    AdaptiveRevalidationPolicy,
    HardFreezeWrapper,
)

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]

backends = [AMEnBackend.TORCHTT, AMEnBackend.NATIVE]


@pytest.mark.parametrize("device, dtype", test_params)
@pytest.mark.parametrize("backend", backends)
def test_amen_solver(device, dtype, backend):
    torch.manual_seed(42)

    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create random matrix and solution
    A = tntt.random(
        [(4, 4), (5, 5), (6, 6), (3, 3)], [1, 2, 3, 2, 1], dtype=dtype, device=device
    )
    b = A @ tntt.random([4, 5, 6, 3], [1, 3, 2, 2, 1], dtype=dtype, device=device)
    x0 = tntt.ones([4, 5, 6, 3], dtype=dtype, device=device)

    # Run torchTT AMEn solve
    xe = tntt.solvers.amen_solve(A, b, x0=x0, use_cpp=True)

    A = Operator(TTEngine(A.cores))
    x0 = State(TTEngine(x0.cores))
    b = State(TTEngine(b.cores))
    ls = LinearSystem(A, source=Source(b))
    ls.state = x0

    # Run AMEnSolver
    solver = AMEnSolver(backend=backend)
    solver.solve(ls)

    # Get the solution vector
    xa = tntt.TT([core.squeeze(2) for core in ls.state.as_tt().cores])
    assert (
        xa - xe
    ).norm() / xe.norm() < 6e-4  # They won't be the exact same because of random enrichment


def _diag_dominant_system(d, n, rank, dtype, device, seed):
    """Well-conditioned enough that both backends actually converge under a modest sweep
    budget -- a purely random TT operator doesn't (see
    benchmarks/amen_scaling_benchmark.py's `random_system` for the same construction and
    why it's used instead of plain `tntt.random`)."""
    torch.manual_seed(seed)
    op_ranks = [1] + [rank] * (d - 1) + [1]
    pert = tntt.random([(n, n)] * d, op_ranks, dtype=dtype).to(device)
    scales = torch.logspace(0, 0.5, d)
    A = pert * 0.01
    for i in range(d):
        A = A + scales[i] * tntt.eye([n] * d, dtype=dtype).to(device)
    A = A.round(1e-13)
    x_true = tntt.random([n] * d, op_ranks, dtype=dtype).to(device)
    b = (A @ x_true).round(1e-13)
    return A, b


@pytest.mark.parametrize("device, dtype", test_params)
@pytest.mark.parametrize(
    "preconditioner", [AMEnPreconditioner.LOCAL_C_PREC, AMEnPreconditioner.LOCAL_R_PREC]
)
def test_amen_solver_local_preconditioner_matches_torchtt(
    device, dtype, preconditioner
):
    """NATIVE's local (per-core) preconditioner (LOCAL_C_PREC, LOCAL_R_PREC)
    should match TORCHTT's own local preconditioner on the same well-conditioned
    problem -- a direct, already-available correctness oracle instead of a
    from-scratch derivation."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    A, b = _diag_dominant_system(3, 8, 2, dtype, device, seed=0)

    results = {}
    for backend in backends:
        A_eng = Operator(TTEngine(A.cores))
        x0_eng = State(TTEngine(tntt.ones(A.N, dtype=dtype, device=device).cores))
        b_eng = State(TTEngine(b.cores))
        ls = LinearSystem(A_eng, source=Source(b_eng))
        ls.state = x0_eng

        solver = AMEnSolver(backend=backend, prec=preconditioner, nswp=15)
        solver.solve(ls)
        x_result = tntt.TT([c.squeeze(2) for c in ls.state.as_tt().cores])
        results[backend] = ((A @ x_result - b).norm() / b.norm()).item()

    tol = 1e-4 if dtype == torch.float32 else 1e-8
    for backend, res in results.items():
        assert (
            res < tol
        ), f"{backend} with preconditioner={preconditioner} residual {res:.3e}"


def test_amen_solver_preserve_moments_wires_through_presolve_and_solve():
    """preserve_moments=True on AMEnSolver should protect a LinearSystem's
    moment_projector through BOTH presolve()'s RHS round (local_solver.cpp) and
    AMEnSolver.solve()'s own solution round -- not just the standalone round_conserved()
    primitive (see test_round_conserved.py).

    Compare a tight (relaxation=1, i.e. no widening) vs. a deliberately loose
    moment_remainder_relaxation from the same problem/warm start: the projected
    moment should agree either way, since it's supposed to be protected
    regardless of remainder_eps. (Not asserting the two solutions differ here --
    round_conserved()'s own aggressive-rounding effect is already covered by
    test_round_conserved.py on a problem sized to actually have "excess" rank to
    shed; this toy diag-dominant system's true solution rank is small enough
    that round_conserved's tight final consolidation pass can legitimately
    reproduce the same minimal representation either way.)
    """
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    A, b = _diag_dominant_system(d, n, r, dtype, device, seed=3)
    torch.manual_seed(4)
    x0 = tntt.random([n] * d, [1, r, r, 1], dtype=dtype).to(device)

    # Simple rank-1 projector on the first mode, independent of any physics
    # -- sufficient to exercise the wiring end-to-end.
    v = torch.rand(n, dtype=dtype)
    v = v / v.norm()
    proj_cores = [torch.outer(v, v).reshape(1, n, n, 1)]
    for _ in range(d - 1):
        proj_cores.append(torch.eye(n, dtype=dtype).reshape(1, n, n, 1))
    P = Operator(TTEngine(proj_cores))

    def _run(relaxation):
        ls = LinearSystem(
            Operator(TTEngine(A.cores)),
            source=Source(State(TTEngine(b.cores))),
            moment_projector=P,
        )
        ls.state = State(TTEngine(x0.cores))
        solver = AMEnSolver(
            nswp=6,
            preserve_moments=True,
            moment_remainder_relaxation=relaxation,
            moment_eps=1e-12,
        )
        solver.solve(ls)
        return ls.state

    # AMEnSolver's default eps is 1e-10, so relaxation=1 (tight) matches the
    # pre-existing eps exactly, and relaxation=5e8 (loose) widens the
    # remainder's own tolerance to ~5e-2 -- decoupled from, but still scaled
    # relative to, whatever eps_forcing has driven get_eps() to.
    x_tight = _run(1.0)
    x_loose = _run(5e8)

    # The protected moment agrees regardless of remainder_eps.
    m_tight = mv(P, x_tight)
    m_loose = mv(P, x_loose)
    rel = (m_tight - m_loose).as_tt().norm() / m_tight.as_tt().norm()
    assert rel < 1e-6


def test_amen_solver_preserve_moments_default_off_is_unchanged():
    """preserve_moments defaults to False -- attaching a moment_projector to the
    LinearSystem without opting in must not change AMEnSolver's behavior in any way that
    matters (well below the solver's own eps=1e-10 rounding tolerance -- not bit-exact,
    since TT-round's per-core truncated SVD is a discontinuous function of its input
    near the truncation threshold, so an O(eps) gap is expected even between two runs
    with no real behavioral difference at all)."""
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    A, b = _diag_dominant_system(d, n, r, dtype, device, seed=3)
    torch.manual_seed(5)
    x0 = tntt.random([n] * d, [1, r, r, 1], dtype=dtype).to(device)

    v = torch.rand(n, dtype=dtype)
    v = v / v.norm()
    proj_cores = [torch.outer(v, v).reshape(1, n, n, 1)]
    for _ in range(d - 1):
        proj_cores.append(torch.eye(n, dtype=dtype).reshape(1, n, n, 1))
    P = Operator(TTEngine(proj_cores))

    ls_with_proj = LinearSystem(
        Operator(TTEngine(A.cores)),
        source=Source(State(TTEngine(b.cores))),
        moment_projector=P,
    )
    ls_with_proj.state = State(TTEngine(x0.cores))

    ls_without_proj = LinearSystem(
        Operator(TTEngine(A.cores)), source=Source(State(TTEngine(b.cores)))
    )
    ls_without_proj.state = State(TTEngine(x0.cores))

    # AMEn's enrichment step draws random kick vectors from the global RNG
    # (see amen_sweep.cpp), so the two solve() calls must be reseeded
    # identically -- otherwise they draw different random enrichment
    # directions and the comparison below is comparing noise, not behavior.
    solver = AMEnSolver(nswp=5)
    torch.manual_seed(7)
    solver.solve(ls_with_proj)
    torch.manual_seed(7)
    solver.solve(ls_without_proj)

    # Not machine epsilon: the solver's own eps=1e-10 rounding tolerance is a
    # truncated-SVD threshold, which is discontinuous in its input, so an
    # ordinary sub-eps floating-point gap between the two runs (e.g. from
    # threaded-BLAS reduction-order jitter in QR, unrelated to any real
    # behavioral difference) can be amplified up to O(eps) per truncation
    # decision, compounded across d=3 cores and nswp=5 sweeps. 1e-5 comfortably
    # clears that noise floor while staying far below the O(1) scale a genuine
    # behavioral difference (e.g. from the projector leaking through) would
    # produce.
    assert (ls_with_proj.state - ls_without_proj.state).as_tt().norm() < 1e-5


def test_amen_solver_gmres_mixed_precision_converges():
    """AMEnNativeOptions(gmres_mixed_precision=True) should wire through AMEnSolver ->
    amen_solve_dispatch -> amen_sweep -> gmres_solve and still converge to a reasonable
    residual on a real (diag-dominant) problem -- CUDA-only, since the option is a no-op
    on CPU tensors (see test_gmres_local.py's unit-level coverage for the primitive
    itself)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device, dtype = "cuda", torch.float64
    d, n, r = 3, 8, 2

    A, b = _diag_dominant_system(d, n, r, dtype, device, seed=6)

    A_eng = Operator(TTEngine(A.cores))
    x0_eng = State(TTEngine(tntt.ones(A.N, dtype=dtype, device=device).cores))
    b_eng = State(TTEngine(b.cores))
    ls = LinearSystem(A_eng, source=Source(b_eng))
    ls.state = x0_eng

    solver = AMEnSolver(
        nswp=15, native_opts=AMEnNativeOptions(gmres_mixed_precision=True)
    )
    solver.solve(ls)

    x_result = tntt.TT([c.squeeze(2) for c in ls.state.as_tt().cores])
    res = ((A @ x_result - b).norm() / b.norm()).item()
    assert res < 1e-6


def test_amen_solver_gmres_mixed_precision_default_off_is_unchanged():
    """gmres_mixed_precision defaults to False -- passing it explicitly False must
    produce a byte-for-byte identical solve to the default AMEnNativeOptions()."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device, dtype = "cuda", torch.float64
    d, n, r = 3, 8, 2

    A, b = _diag_dominant_system(d, n, r, dtype, device, seed=7)

    # AMEn's enrichment step draws torch.randn(...) for random enrichment
    # columns -- reseed before each solve so the two runs consume identical
    # RNG state and are actually comparable bit-for-bit (the difference this
    # is meant to isolate is only the native_opts flag).
    def _run(native_opts):
        torch.manual_seed(8)
        A_eng = Operator(TTEngine(A.cores))
        x0_eng = State(TTEngine(tntt.ones(A.N, dtype=dtype, device=device).cores))
        b_eng = State(TTEngine(b.cores))
        ls = LinearSystem(A_eng, source=Source(b_eng))
        ls.state = x0_eng
        solver = AMEnSolver(nswp=8, native_opts=native_opts)
        solver.solve(ls)
        return ls.state

    x_default = _run(AMEnNativeOptions())
    x_explicit_off = _run(AMEnNativeOptions(gmres_mixed_precision=False))

    assert (x_default - x_explicit_off).as_tt().norm() < 1e-14


def test_amen_solver_eps_max_caps_forced_tolerance():
    """eps_ = max(eps_floor_, min(eps_max_, eps_forcing_ * min_error_)).

    min_error_ starts at 1, so without a cap a forcing factor above ~1 asks the
    first solves for a relative truncation tolerance near or above 1, which
    rounds the solution away. With a cap, a large forcing factor is held at
    eps_max until eps_forcing * min_error falls below it, and the floor still
    applies at the bottom.
    """
    floor, cap = 1e-6, 0.1
    solver = AMEnSolver(eps=floor, eps_forcing=10.0, eps_max=cap)
    assert solver.eps_max == cap

    # (reported error, expected eps after the update)
    for error, expected in [
        (0.8, cap),  # 10 * 0.8 = 8, capped at 0.1
        (2.0, cap),  # min_error_ does not increase, still capped
        (5e-3, 5e-2),  # 10 * 5e-3 below the cap: forcing applies
        (1e-9, floor),  # 10 * 1e-9 below the floor: floor applies
    ]:
        solver.update_convergence_criteria(error)
        assert solver.eps == pytest.approx(expected)


def test_amen_solver_eps_max_default_is_uncapped():
    """The default eps_max is +inf, so forcing is unchanged from before."""
    solver = AMEnSolver(eps=1e-6, eps_forcing=10.0)
    assert solver.eps_max == float("inf")
    solver.update_convergence_criteria(0.8)
    assert solver.eps == pytest.approx(8.0)


def test_amen_solver_eps_max_below_floor_rejected():
    """A cap below the floor is contradictory and must be rejected."""
    with pytest.raises(RuntimeError):
        AMEnSolver(eps=1e-3, eps_max=1e-4)


def test_amen_solver_enrichment_policy_rejects_torchtt_backend():
    """An EnrichmentPolicy relies on amen_sweep.cpp's `enrichment_disabled` zero-rank
    code path, which only exists for AMEnBackend.NATIVE -- the vendored torchTT backend
    crashes (ambiguous 0-element reshape) if enrichment is ever forced to zero rank.

    Must be rejected up front.
    """
    with pytest.raises(RuntimeError):
        AMEnSolver(
            backend=AMEnBackend.TORCHTT,
            enrichment_policy=StaticFreezePolicy(freeze_eps=1e-3),
        )


def test_amen_solver_rank_freeze():
    """StaticFreezePolicy: once the solver's own adaptively-tightening eps
    (see AMEnSolver.update_convergence_criteria) drops to or below
    freeze_eps, it should permanently switch to pure ALS (zero enrichment)
    for every later solve() call on this instance, and per-bond rank should
    never grow again from that point on -- see memory
    project_c5g7_amen_options_sweep's "grow-then-freeze" question."""
    torch.manual_seed(0)
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    A, b = _diag_dominant_system(d, n, r, dtype, device, seed=1)
    x0 = tntt.random([n] * d, [1, r, r, 1], dtype=dtype).to(device)

    A_eng = Operator(TTEngine(A.cores))
    b_eng = State(TTEngine(b.cores))
    x0_eng = State(TTEngine(x0.cores))
    ls = LinearSystem(A_eng, source=Source(b_eng))
    ls.state = x0_eng

    freeze_eps = 1e-3
    solver = AMEnSolver(
        eps=1e-8,
        eps_forcing=1.0,  # eps_ = max(eps_floor_, min_error_ so far)
        nswp=5,
        enrichment_policy=StaticFreezePolicy(freeze_eps=freeze_eps),
    )
    assert not solver.is_rank_frozen()

    errors = [1.0, 0.5, 0.1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7]
    frozen_before_solve = []
    ranks_over_time = []
    residuals_over_time = []
    for err in errors:
        frozen_before_solve.append(solver.is_rank_frozen())
        solver.solve(ls)
        ranks_over_time.append(ls.state.as_tt().ranks)
        x_now = tntt.TT([c.squeeze(2) for c in ls.state.as_tt().cores])
        residuals_over_time.append(((A @ x_now - b).norm() / b.norm()).item())
        solver.update_convergence_criteria(err)

    assert not any(frozen_before_solve[:4])
    assert any(frozen_before_solve)
    first_frozen = frozen_before_solve.index(True)

    # Once frozen, every bond's rank must be non-increasing sweep over
    # sweep -- the SVD truncation loop inside amen_sweep can only shrink or
    # hold rank when enrichment is disabled, never grow it, regardless of
    # how the RHS evolves (see the "Correction" note in the rank-freeze
    # plan).
    for prev, curr in zip(
        ranks_over_time[first_frozen:], ranks_over_time[first_frozen + 1 :]
    ):
        assert all(rc <= rp for rc, rp in zip(curr, prev))

    # Pure ALS should still be doing useful work: residual keeps improving
    # after freezing, not stalled. Once both sides of the comparison are
    # already at float64's noise floor (~1e-15), which of two values is
    # marginally smaller is meaningless -- an absolute tolerance at that
    # scale keeps this a real regression check (e.g. would still catch a
    # stall at 1e-3) without being sensitive to which factorization method
    # (QR vs. SVD) happens to be in use for the frozen-rank path.
    assert residuals_over_time[-1] < residuals_over_time[first_frozen] + 1e-13
    assert residuals_over_time[-1] < 1e-4


def test_amen_solver_adaptive_revalidation_toggles_via_update_convergence_criteria():
    """AdaptiveRevalidationPolicy is active from the very first call (no
    threshold) and widens the enrich/ALS cycle once rank_metric stops
    growing -- driving it with a synthetic, flat rank_metric sequence
    (bypassing the need for a real solve, since should_enrich only consumes
    eps/error/rank_metric) should eventually produce some is_rank_frozen()
    == True calls, then go back to consistently enriching once real growth
    is reported again."""
    solver = AMEnSolver(
        eps=1e-8,
        eps_forcing=1.0,
        nswp=1,
        enrichment_policy=AdaptiveRevalidationPolicy(
            initial_period=1,
            probe_iterations=1,
            growth_factor=2.0,
            max_period=8,
            growth_tolerance=0.01,
        ),
    )
    assert not solver.is_rank_frozen()

    # Flat rank_metric -- the cycle should widen, producing some frozen
    # (cheap-ALS) calls once it does.
    frozen_pattern = []
    for _ in range(20):
        solver.update_convergence_criteria(1.0, 100.0)
        frozen_pattern.append(solver.is_rank_frozen())
    assert any(frozen_pattern)

    # Steady growth every call thereafter -- should return to consistently
    # enriching (not frozen).
    metric = 100.0
    for _ in range(6):
        metric *= 1.5
        solver.update_convergence_criteria(1.0, metric)
    assert not solver.is_rank_frozen()


def test_amen_solver_hard_freeze_wrapper_composes_adaptive_and_static():
    """HardFreezeWrapper(AdaptiveRevalidationPolicy(...), freeze_eps): active, self-
    pacing enrichment while eps > freeze_eps, then a permanent freeze once eps drops to
    or below freeze_eps -- matching StaticFreezePolicy's already-proven late-stage
    behavior (see test_amen_solver_rank_freeze).

    Composing rather than duplicating gets both properties at once.
    """
    torch.manual_seed(0)
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    A, b = _diag_dominant_system(d, n, r, dtype, device, seed=1)
    x0 = tntt.random([n] * d, [1, r, r, 1], dtype=dtype).to(device)

    A_eng = Operator(TTEngine(A.cores))
    b_eng = State(TTEngine(b.cores))
    x0_eng = State(TTEngine(x0.cores))
    ls = LinearSystem(A_eng, source=Source(b_eng))
    ls.state = x0_eng

    freeze_eps = 1e-3
    solver = AMEnSolver(
        eps=1e-8,
        eps_forcing=1.0,  # eps_ = max(eps_floor_, min_error_ so far)
        nswp=5,
        enrichment_policy=HardFreezeWrapper(
            inner=AdaptiveRevalidationPolicy(initial_period=1, probe_iterations=1),
            freeze_eps=freeze_eps,
        ),
    )
    assert not solver.is_rank_frozen()

    errors = [1.0, 0.5, 0.1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7]
    ranks_over_time = []
    for err in errors:
        solver.solve(ls)
        ranks_over_time.append(ls.state.as_tt().ranks)
        rank_metric = float(sum(ls.state.as_tt().ranks))
        solver.update_convergence_criteria(err, rank_metric)

    # By the last couple of errors (well below freeze_eps), the hard freeze
    # must have engaged -- rank can no longer grow, matching
    # StaticFreezePolicy's own invariant once truly frozen.
    assert solver.is_rank_frozen()
    assert solver.enrichment_policy.has_frozen()
    assert all(rc <= rp for rc, rp in zip(ranks_over_time[-1], ranks_over_time[-2]))
