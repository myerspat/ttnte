import pytest
import torch

from ttnte.linalg import dense_gmres


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
