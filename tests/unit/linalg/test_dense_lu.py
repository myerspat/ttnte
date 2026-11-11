import pytest
import torch

from ttnte.linalg import dense_lu_factor, dense_lu_solve


@pytest.mark.parametrize("n", [1, 4, 9, 27, 40])
def test_matches_torch_solve(n):
    gen = torch.Generator().manual_seed(n)
    a = torch.randn(6, n, n, generator=gen, dtype=torch.float64) + n * torch.eye(
        n, dtype=torch.float64
    )
    b = torch.randn(6, n, generator=gen, dtype=torch.float64)
    lu, ipiv = dense_lu_factor(a)
    x = dense_lu_solve(lu, ipiv, b)
    ref = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(x, ref, rtol=1e-12, atol=1e-12)


def test_pivoting_required():
    a = torch.tensor([[[0.0, 2.0], [3.0, 1.0]]], dtype=torch.float64)
    lu, ipiv = dense_lu_factor(a)
    assert ipiv[0, 0].item() == 1
    x = dense_lu_solve(lu, ipiv, torch.tensor([[2.0, 4.0]], dtype=torch.float64))
    assert torch.allclose(a[0] @ x[0], torch.tensor([2.0, 4.0], dtype=torch.float64))


def test_singular_raises():
    with pytest.raises(RuntimeError):
        dense_lu_factor(torch.zeros(1, 3, 3, dtype=torch.float64))
