import pytest
import torch

from ttnte.linalg import dense_gmres, dense_gmres_progress


def _system(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    A = (
        torch.eye(n, dtype=torch.float64)
        + 0.3 * torch.randn(n, n, generator=gen, dtype=torch.float64) / n**0.5
    )
    b = torch.randn(n, generator=gen, dtype=torch.float64)
    return A, b


@pytest.mark.parametrize("restart", [5, 30, 200])
def test_converges_to_direct_solve(restart):
    A, b = _system(60)
    x, applies, residual, converged = dense_gmres(
        A, b, restart=restart, tol=1e-12, max_applies=5000
    )
    assert converged and residual <= 1e-12
    assert torch.allclose(x, torch.linalg.solve(A, b), rtol=1e-9, atol=1e-10)


def test_warm_start_and_zero_rhs():
    A, b = _system(30, seed=1)
    exact = torch.linalg.solve(A, b)
    _, applies, _, converged = dense_gmres(A, b, x0=exact, tol=1e-10)
    assert converged and applies == 1  # only the initial residual check
    x, _, _, converged = dense_gmres(A, torch.zeros(30, dtype=torch.float64))
    assert converged and torch.count_nonzero(x) == 0


def test_respects_application_budget():
    A, b = _system(80, seed=2)
    _, applies, residual, converged = dense_gmres(
        A, b, restart=3, tol=1e-14, max_applies=7
    )
    assert not converged and applies <= 7 and residual > 1e-14


# ---------------------------------------------------------------------------
# The algorithm, not only the answer: GMRES's k-th iterate is the unique minimizer
# of ||b - A x|| over x0 + K_k(A, r0), computed here independently (Householder QR
# of the Krylov basis and a least-squares solve), on non-normal upwind-like
# matrices of the kind the sweeper solves (I - T S, T an upwind sweep inverse).


def _upwind(n, seed=0, c=0.9, coupling=0.95):
    """I - c L^{-1} S: L lower bidiagonal (a 1-D upwind sweep, L^{-1} strongly
    non-normal), S a random nonnegative scattering-like matrix."""
    gen = torch.Generator().manual_seed(seed)
    L = torch.eye(n, dtype=torch.float64) - coupling * torch.diag(
        torch.ones(n - 1, dtype=torch.float64), -1
    )
    S = torch.rand(n, n, generator=gen, dtype=torch.float64) / n
    A = torch.eye(n, dtype=torch.float64) - c * torch.linalg.solve(L, S)
    b = torch.randn(n, generator=gen, dtype=torch.float64)
    return A, b


def _bidiagonal(n, coupling=0.95):
    """The upwind sweep itself: one eigenvalue, Jordan-like, slow GMRES."""
    return torch.eye(n, dtype=torch.float64) - coupling * torch.diag(
        torch.ones(n - 1, dtype=torch.float64), -1
    )


def _minimizer(A, r0, k):
    """argmin over z in K_k(A, r0) of ||r0 - A z||."""
    Q = (r0 / r0.norm()).unsqueeze(1)
    for _ in range(k - 1):
        w = A @ Q[:, -1]
        Q, _ = torch.linalg.qr(torch.cat([Q, w.unsqueeze(1)], dim=1))
    y = torch.linalg.lstsq(A @ Q, r0.unsqueeze(1)).solution.squeeze(1)
    return Q @ y


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _true_residual(A, b, x):
    return ((b - A @ x).norm() / b.norm()).item()


MATRICES = {
    "upwind I - T S": lambda: _upwind(60),
    "bidiagonal sweep": lambda: (_bidiagonal(40), _upwind(40, seed=3)[1]),
}


@pytest.mark.parametrize("name", list(MATRICES))
@pytest.mark.parametrize("k", [1, 3, 8, 15])
def test_iterate_is_the_krylov_minimizer(name, k):
    """K applications from a zero guess (no product for r0 = b): the minimizer over
    K_k(A, b); the reported residual is that iterate's true residual."""
    A, b = MATRICES[name]()
    x, applies, residual, converged = dense_gmres(
        A, b, restart=50, tol=0.0, max_applies=k
    )
    assert applies == k and not converged
    assert _rel(x, _minimizer(A, b, k)) < 1e-8
    assert residual == pytest.approx(_true_residual(A, b, x), rel=1e-8)


@pytest.mark.parametrize("name", list(MATRICES))
def test_restarted_iterate_is_the_minimizer_per_cycle(name):
    """GMRES(4): a cycle of 4 products, one product for the true residual, then 3
    more: x = x1 + minimizer over K_3(A, b - A x1), x1 the first cycle's."""
    A, b = MATRICES[name]()
    x, applies, residual, _ = dense_gmres(A, b, restart=4, tol=0.0, max_applies=8)
    assert applies == 8
    x1 = _minimizer(A, b, 4)
    ref = x1 + _minimizer(A, b - A @ x1, 3)
    assert _rel(x, ref) < 1e-8
    assert residual == pytest.approx(_true_residual(A, b, x), rel=1e-8)


def test_warm_started_iterate_is_the_minimizer():
    """From x0: one product for r0 = b - A x0, then k: x0 + minimizer over K_k(A, r0)."""
    A, b = _upwind(50, seed=4)
    x0 = torch.randn(
        50, dtype=torch.float64, generator=torch.Generator().manual_seed(5)
    )
    x, applies, _, _ = dense_gmres(A, b, x0=x0, restart=30, tol=0.0, max_applies=7)
    assert applies == 7
    assert _rel(x, x0 + _minimizer(A, b - A @ x0, 6)) < 1e-8


@pytest.mark.parametrize("name", list(MATRICES))
@pytest.mark.parametrize("restart", [5, 20])
def test_reported_residual_is_the_true_residual(name, restart):
    """Convergence is accepted on the Arnoldi estimate: it must match the true
    residual of the returned x (the sweeper's tolerances rely on it)."""
    A, b = MATRICES[name]()
    tol = 1e-8
    x, _, residual, converged = dense_gmres(
        A, b, restart=restart, tol=tol, max_applies=5000
    )
    true = _true_residual(A, b, x)
    assert converged and residual <= tol
    assert true <= tol * (1 + 1e-6)
    assert abs(residual - true) <= 1e-6 * true + 1e-14
    assert torch.allclose(x, torch.linalg.solve(A, b), rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Breakdown: an invariant Krylov space


def test_lucky_breakdown_solves_exactly():
    """B an eigenvector of a non-normal A, or A with 3 distinct eigenvalues: the Krylov
    space is invariant after 1 or 3 products and the solve is exact.

    (V is
    well conditioned: forming V D V^{-1} with an ill-conditioned V leaves A's
    invariant spaces only approximately invariant, above the breakdown threshold.)
    """
    n = 20
    gen = torch.Generator().manual_seed(6)
    V = torch.eye(n, dtype=torch.float64) + 0.05 * torch.randn(
        n, n, generator=gen, dtype=torch.float64
    )
    D = torch.diag(torch.tensor([1.0, 2.5, 4.0] * 6 + [1.0, 2.5], dtype=torch.float64))
    A = V @ D @ torch.linalg.inv(V)
    for b, products in [
        (V[:, 0], 1),
        (torch.randn(n, generator=gen, dtype=torch.float64), 3),
    ]:
        x, applies, residual, converged = dense_gmres(
            A, b, restart=30, tol=0.0, max_applies=100
        )
        assert converged and applies == products
        assert residual < 1e-12 and _true_residual(A, b, x) < 1e-12
    x, applies, _, converged = dense_gmres(
        torch.eye(n, dtype=torch.float64), V[:, 1], tol=0.0
    )
    assert converged and applies == 1 and torch.allclose(x, V[:, 1])
    # Exact arithmetic: a non-normal upper bidiagonal A with A e1 = 2 e1
    A = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 5.0]], dtype=torch.float64
    )
    b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    x, applies, residual, converged = dense_gmres(A, b, tol=0.0)
    assert converged and applies == 1 and residual == 0.0
    assert torch.equal(x, b / 2)


def test_singular_inconsistent_system_is_not_reported_converged():
    """A = diag(1, 2, 0), b = (1, 1, 1): K_2 is invariant and A is singular on it.

    GMRES stops at the least-squares solution (1, 1/2, 0) with residual
    1/sqrt(3) and does not claim convergence (it used to divide by a rounding-
    level pivot, report a residual of 1e-16 and return |x| ~ 1e14). x_3 is free
    (A's null space), so the check is on A x.
    """
    A = torch.diag(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64))
    b = torch.ones(3, dtype=torch.float64)
    x, applies, residual, converged = dense_gmres(A, b, restart=10, tol=1e-12)
    assert not converged and applies <= 3
    assert torch.allclose(A @ x, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))
    assert residual == pytest.approx(3**-0.5, rel=1e-12)
    assert residual == pytest.approx(_true_residual(A, b, x), rel=1e-12)


def test_singular_consistent_system_converges():
    A = torch.diag(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64))
    b = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    x, _, _, converged = dense_gmres(A, b, restart=10, tol=1e-12)
    assert converged
    assert torch.allclose(x, torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64))


# ---------------------------------------------------------------------------
# Distributed vectors (the reduce hook): the vectors split over threads standing in
# for ranks, each running gmres() on its slice with a summing reducer


@pytest.mark.parametrize("parts", [2, 3, 7])
@pytest.mark.parametrize("restart", [5, 30])
def test_distributed_matches_serial(parts, restart):
    A, b = _upwind(61, seed=7)  # 61 rows: uneven slices
    serial = dense_gmres(A, b, restart=restart, tol=1e-10, max_applies=2000)
    split = dense_gmres(A, b, restart=restart, tol=1e-10, max_applies=2000, parts=parts)
    assert split[1] == serial[1] and split[3] == serial[3]
    assert _rel(split[0], serial[0]) < 1e-10
    assert split[2] == pytest.approx(serial[2], rel=1e-6)


def test_distributed_guess_zero_on_some_slices_only():
    """A warm start that is zero on one rank's slice only: every rank must still apply
    the operator for the initial residual (the zero-guess check is reduced), or the
    ranks' collective operations no longer pair up."""
    A, b = _upwind(40, seed=8)
    x0 = torch.zeros(40, dtype=torch.float64)
    x0[30:] = 1.0  # the last of 2 slices only
    serial = dense_gmres(A, b, x0=x0, restart=10, tol=1e-10)
    split = dense_gmres(A, b, x0=x0, restart=10, tol=1e-10, parts=2)
    assert split[1] == serial[1] and split[3]
    assert _rel(split[0], serial[0]) < 1e-10


def test_distributed_singular_stagnation():
    A = torch.diag(torch.tensor([1.0, 2.0, 0.0, 3.0], dtype=torch.float64))
    b = torch.ones(4, dtype=torch.float64)
    x, _, residual, converged = dense_gmres(A, b, restart=10, tol=1e-12, parts=2)
    assert not converged and residual == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# The progress hook (drives the sweeper's relaxed inexact products)


@pytest.mark.parametrize("x0", [None, "warm"])
def test_progress_called_before_every_application(x0):
    """One call before each product, flagged true for the restart's true residual
    (reported as 1, to be applied accurately) and otherwise carrying the current
    Arnoldi estimate: 1 at the start of a cold solve, non-increasing within a
    cycle, ending at or above the returned residual."""
    A, b = _upwind(50, seed=9)
    m = 6
    guess = None if x0 is None else 0.1 * b
    x, applies, residual, converged, calls = dense_gmres_progress(
        A, b, x0=guess, restart=m, tol=1e-9, max_applies=500
    )
    assert converged
    assert len(calls) == applies
    assert [c[2] for c in calls] == list(range(applies))
    flags = [c[1] for c in calls]
    # Cold: m Arnoldi products per cycle, then a true-residual product
    # Warm: the true-residual product first
    period = [False] * m + [True]
    expected = (
        (period * applies)[:applies]
        if x0 is None
        else ([True] + period * applies)[:applies]
    )
    assert flags == expected
    for value, flag, _ in calls:
        if flag:
            assert value == 1.0
    if x0 is None:
        assert calls[0][0] == 1.0
    cycle = []
    for value, flag, _ in calls + [(None, True, None)]:
        if flag:
            assert all(a >= b * (1 - 1e-12) for a, b in zip(cycle, cycle[1:]))
            cycle = []
        else:
            cycle.append(value)
    assert calls[-1][0] >= residual
