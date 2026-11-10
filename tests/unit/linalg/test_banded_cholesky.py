"""Banded Cholesky (LAPACK dpbtrf, dtbmv, dtbsv), the factor of the symmetric positive
definite mass matrix that the cross uses for its weighted inner product on patches above
the dense-factor limit, checked against torch's dense Cholesky and triangular solve."""

import pytest
import torch

from ttnte.linalg import (
    banded_cholesky_apply_upper,
    banded_cholesky_factor,
    banded_cholesky_solve_upper,
    symmetric_band_from_pairs,
)


def _spd_band(n, kd, seed=0):
    """Random symmetric band matrix made positive definite by strict diagonal
    dominance."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=gen, dtype=torch.float64)
    i = torch.arange(n)
    a = torch.where((i[:, None] - i[None, :]).abs() <= kd, a + a.T, 0.0)
    a.diagonal().copy_(a.abs().sum(dim=1) + 1.0)
    return a


def _to_lower_band(a, kd):
    """(n, kd + 1) storage: row j holds A(j + i, j) at column i."""
    n = a.shape[0]
    ab = torch.zeros(n, kd + 1, dtype=torch.float64)
    for j in range(n):
        for i in range(min(kd + 1, n - j)):
            ab[j, i] = a[j + i, j]
    return ab


def _from_lower_band(ab, kd):
    n = ab.shape[0]
    L = torch.zeros(n, n, dtype=torch.float64)
    for j in range(n):
        for i in range(min(kd + 1, n - j)):
            L[j + i, j] = ab[j, i]
    return L


CASES = [(1, 0), (9, 0), (12, 1), (30, 3), (50, 7), (8, 7)]


@pytest.mark.parametrize("n, kd", CASES)
def test_factor_matches_torch_cholesky(n, kd):
    a = _spd_band(n, kd)
    L = _from_lower_band(banded_cholesky_factor(_to_lower_band(a, kd), kd), kd)
    ref = torch.linalg.cholesky(a)
    assert torch.allclose(L, ref, rtol=1e-12, atol=1e-12)
    assert torch.allclose(L @ L.T, a, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n, kd", CASES)
def test_apply_and_solve_upper(n, kd):
    """apply_upper is x -> L^T x and solve_upper inverts it (the M^(1/2) weighting and
    unweighting of the cross)."""
    a = _spd_band(n, kd, seed=1)
    factor = banded_cholesky_factor(_to_lower_band(a, kd), kd)
    L = torch.linalg.cholesky(a)
    x = torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(
        banded_cholesky_apply_upper(factor, kd, x), L.T @ x, rtol=1e-12
    )
    ref = torch.linalg.solve_triangular(L.T, x.unsqueeze(-1), upper=True).squeeze(-1)
    y = banded_cholesky_solve_upper(factor, kd, x)
    assert ((y - ref).norm() / ref.norm()).item() < 1e-12
    assert torch.allclose(banded_cholesky_apply_upper(factor, kd, y), x, rtol=1e-12)


def test_not_positive_definite_raises():
    a = _spd_band(10, 2)
    a[4, 4] = -1.0
    with pytest.raises(RuntimeError):
        banded_cholesky_factor(_to_lower_band(a, 2), 2)


def test_symmetric_band_from_pairs():
    """Pair lists (as the assembled mass matrix arrives) pack into the same lower band
    storage; pairs above the diagonal are ignored and repeated pairs add."""
    a = _spd_band(15, 3, seed=3)
    rows, cols = torch.nonzero(a, as_tuple=True)
    vals = a[rows, cols]
    # Split one diagonal entry over two pairs
    rows = torch.cat([rows, torch.tensor([5])])
    cols = torch.cat([cols, torch.tensor([5])])
    vals = torch.cat([vals, torch.tensor([1.5], dtype=torch.float64)])
    a[5, 5] += 1.5
    ab, kd = symmetric_band_from_pairs(rows, cols, vals, 15)
    assert kd == 3
    assert torch.equal(ab, _to_lower_band(a, 3))
