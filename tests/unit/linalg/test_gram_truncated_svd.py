import pytest
import torch

from ttnte.linalg import gram_truncated_svd, truncated_svd


def _matrix(m, n, decay, seed=0):
    """(m, n) matrix with geometrically decaying singular values."""
    g = torch.Generator().manual_seed(seed)
    k = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, k, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, k, generator=g, dtype=torch.float64))
    s = decay ** torch.arange(k, dtype=torch.float64)
    return (U * s) @ V.T


@pytest.mark.parametrize("shape", [(4000, 40), (40, 4000)])
@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6])
def test_matches_truncated_svd(shape, tol):
    A = _matrix(*shape, decay=0.7)
    U, S, Vh, delta, rest = gram_truncated_svd(A, tol, with_normalize=True)
    _, S0, _, delta0, _ = truncated_svd(A, tol, with_normalize=True)
    assert S.numel() == S0.numel()
    assert torch.allclose(S, S0, rtol=1e-6, atol=1e-10 * S0[0].item())
    assert abs(delta - delta0) <= 1e-12 * delta0
    err = torch.linalg.norm(A - (U * S) @ Vh) / torch.linalg.norm(A)
    assert err <= tol
    assert rest >= 0.0


def test_max_rank():
    A = _matrix(3000, 30, decay=0.9)
    U, S, Vh, *_ = gram_truncated_svd(A, 1e-12, max_rank=5)
    assert U.shape == (3000, 5) and S.shape == (5,) and Vh.shape == (5, 30)


def test_square_and_tight_fall_back():
    # Near-square, and a threshold below the Gram floor: the QR path is used,
    # so the result is exactly truncated_svd's
    for A, tol in [(_matrix(60, 50, 0.7), 1e-4), (_matrix(4000, 40, 0.7), 1e-9)]:
        U, S, Vh, *_ = gram_truncated_svd(A, tol, with_normalize=True)
        U0, S0, Vh0, *_ = truncated_svd(A, tol, with_normalize=True)
        assert torch.equal(S, S0) and torch.equal(U, U0) and torch.equal(Vh, Vh0)
