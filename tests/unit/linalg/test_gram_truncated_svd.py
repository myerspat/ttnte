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


@pytest.mark.parametrize(
    "shape", [(10007, 13), (13, 10007), (64, 64), (5000, 1), (200, 7)]
)
def test_blocked_gram(shape):
    """The blocked Gram equals the plain one, remainder rows or columns included, for
    tall, wide and square inputs."""
    from ttnte.linalg import blocked_gram

    gen = torch.Generator().manual_seed(1)
    A = torch.randn(*shape, dtype=torch.float64, generator=gen)
    ref = A.T @ A if shape[0] >= shape[1] else A @ A.T
    G = blocked_gram(A)
    assert torch.allclose(G, ref, rtol=1e-12, atol=1e-10 * ref.abs().max().item())


@pytest.mark.parametrize("shape", [(6000, 30), (30, 6000)])
@pytest.mark.parametrize("tol", [1e-4, 1e-6, 2e-7])
def test_long_factor_orthonormal(shape, tol):
    """Near the Gram floor the long factor is re-orthogonalized: orthonormal
    to round-off, not to eps (sigma_1 / sigma_r)^2."""
    A = _matrix(*shape, decay=0.6)
    U, S, Vh, *_ = gram_truncated_svd(A, tol, with_normalize=True)
    long = U if shape[0] > shape[1] else Vh.T
    eye = torch.eye(long.shape[1], dtype=torch.float64)
    assert (long.T @ long - eye).abs().max().item() < 1e-10
    err = torch.linalg.norm(A - (U * S) @ Vh) / torch.linalg.norm(A)
    assert err <= tol


@pytest.mark.parametrize("svd", [truncated_svd, gram_truncated_svd])
def test_rest_counts_discarded_energy(svd):
    """rest = sqrt(max(delta^2 - discarded energy, 0)), also when max_rank
    cuts deeper than delta."""
    A = _matrix(3000, 30, decay=0.8)
    s = torch.linalg.svdvals(A)
    for tol, max_rank in [(1e-3, 30), (1e-3, 5)]:
        U, S, Vh, delta, rest = svd(A, tol, max_rank, True)
        r = S.numel()
        discarded = (s[r:] ** 2).sum().item()
        expect = max(delta**2 - discarded, 0.0) ** 0.5
        assert abs(rest - expect) <= 1e-8 * delta


@pytest.mark.parametrize("shape", [(40, 4000), (4000, 40), (60, 50)])
def test_truncated_svd_shapes(shape):
    """truncated_svd's wide (QR of A^T), tall and square branches all give an
    orthonormal factorization within the tolerance."""
    A = _matrix(*shape, decay=0.7)
    U, S, Vh, *_ = truncated_svd(A, 1e-6, with_normalize=True)
    err = torch.linalg.norm(A - (U * S) @ Vh) / torch.linalg.norm(A)
    assert err <= 1e-6
    eye = torch.eye(S.numel(), dtype=torch.float64)
    assert torch.allclose(U.T @ U, eye, atol=1e-12)
    assert torch.allclose(Vh @ Vh.T, eye, atol=1e-12)
