import pytest
import torch

from ttnte.linalg import bandwidth, banded_lu_factor, banded_lu_solve, dense_to_band


def _random_banded(batch, n, kl, ku, seed=0):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, n, n, generator=gen, dtype=torch.float64)
    i = torch.arange(n)
    mask = ((i[:, None] - i[None, :]) <= kl) & ((i[None, :] - i[:, None]) <= ku)
    # Shift the diagonal so the random triangular cases stay well conditioned
    return a * mask + n * torch.eye(n, dtype=torch.float64)


@pytest.mark.parametrize(
    "n, kl, ku", [(1, 0, 0), (7, 0, 0), (12, 3, 2), (40, 7, 7), (25, 0, 5), (25, 5, 0)]
)
def test_matches_dense_solve(n, kl, ku):
    """Banded LU with partial pivoting reproduces dense solves, including matrices whose
    leading entries force row interchanges."""
    a = _random_banded(4, n, kl, ku)
    b = torch.randn(4, n, dtype=torch.float64)
    ab, ipiv = banded_lu_factor(dense_to_band(a, kl, ku), kl, ku)
    x = banded_lu_solve(ab, ipiv, kl, ku, b)
    ref = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(x, ref, rtol=1e-10, atol=1e-10)


def test_pivoting_required():
    """A zero leading diagonal entry needs a row interchange."""
    a = torch.tensor(
        [[[0.0, 1.0, 0.0], [2.0, 1.0, 1.0], [0.0, 3.0, 1.0]]], dtype=torch.float64
    )
    kl, ku = bandwidth(a)
    assert (kl, ku) == (1, 1)
    ab, ipiv = banded_lu_factor(dense_to_band(a, kl, ku), kl, ku)
    assert ipiv[0, 0].item() == 1
    b = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    x = banded_lu_solve(ab, ipiv, kl, ku, b)
    assert torch.allclose(a[0] @ x[0], b[0])


def test_singular_raises():
    a = torch.zeros(1, 3, 3, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        banded_lu_factor(dense_to_band(a, 1, 1), 1, 1)
