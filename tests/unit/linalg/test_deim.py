"""DEIM / Q-DEIM row selection and the quadrature-weighted orthonormal basis.

Han et al. (2026)'s pieces, against the Python reference of the earlier
comparison (ttnte-studies/sweep-cross/han.py, reproduced here) on random bases.
"""

import math

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from ttnte.linalg import (
    deim_select,
    lebesgue_constant,
    qdeim_select,
    weighted_orthonormal,
)

SHAPES = [(12, 1), (40, 5), (200, 17), (64, 64)]


def _basis(n, r, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(n, r, generator=gen, dtype=torch.float64))[0]


def _deim_ref(V):
    """han.py's deim, the paper's DEIM selection.

    Algorithm B.1 of Han et al. (2026).
    """
    I = [int(np.argmax(np.abs(V[:, 0])))]
    for j in range(1, V.shape[1]):
        c = np.linalg.solve(V[I, :j], V[I, j])
        res = V[:, j] - V[:, :j] @ c
        I.append(int(np.argmax(np.abs(res))))
    return I


def _qdeim_ref(V):
    """han.py's qdeim: the first r pivots of a column-pivoted QR of V^T."""
    return list(sla.qr(V.T, mode="r", pivoting=True)[1][: V.shape[1]])


@pytest.mark.parametrize("n, r", SHAPES)
@pytest.mark.parametrize("seed", [0, 1])
def test_deim_matches_reference(n, r, seed):
    V = _basis(n, r, seed)
    assert deim_select(V) == _deim_ref(V.numpy())


@pytest.mark.parametrize("n, r", SHAPES)
@pytest.mark.parametrize("seed", [0, 1])
def test_qdeim_matches_reference(n, r, seed):
    V = _basis(n, r, seed)
    assert qdeim_select(V) == _qdeim_ref(V.numpy())


@pytest.mark.parametrize("select", [deim_select, qdeim_select])
def test_interpolation_is_exact_on_the_span(select):
    """Distinct rows, and for y = V a the interpolant V V(I,:)^{-1} y(I) is y."""
    V = _basis(80, 9, 3)
    I = select(V)
    assert len(set(I)) == len(I) == 9
    a = torch.randn(9, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    y = V @ a
    rows = torch.tensor(I)
    y_interp = V @ torch.linalg.solve(V[rows], y[rows])
    assert ((y_interp - y).norm() / y.norm()).item() < 1e-12


def test_weighted_orthonormal():
    """L = V R with V^T diag(w) V = I and R upper triangular.

    The weighted orthonormality of eq. 3.5 in Han et al. (2026).
    """
    gen = torch.Generator().manual_seed(5)
    L = torch.randn(50, 7, dtype=torch.float64, generator=gen)
    w = torch.rand(50, dtype=torch.float64, generator=gen) + 0.1
    V, R = weighted_orthonormal(L, w)
    assert (
        V.T @ (w[:, None] * V) - torch.eye(7, dtype=torch.float64)
    ).abs().max() < 1e-13
    assert ((V @ R - L).norm() / L.norm()).item() < 1e-13
    assert torch.equal(R, torch.triu(R))


def test_lebesgue_constant():
    """The 2-norm of the interpolation projector P g = V V(I, :)^{-1} g(I), formed
    explicitly (not the orthonormal-basis formula the function uses), and at least
    1."""
    n, r = 60, 6
    V = _basis(n, r, 6) @ torch.randn(
        r, r, dtype=torch.float64, generator=torch.Generator().manual_seed(7)
    )
    I = deim_select(V)
    P = torch.zeros(n, n, dtype=torch.float64)
    P[:, I] = V @ torch.linalg.inv(V[I])
    ref = torch.linalg.matrix_norm(P, 2).item()
    value = lebesgue_constant(V, I)
    assert value == pytest.approx(ref, rel=1e-10) and value >= 1.0


def test_weighted_lebesgue_constant():
    """With weights: the norm of the interpolation projector in ||W^1/2 g||."""
    gen = torch.Generator().manual_seed(8)
    n, r = 50, 5
    V = torch.randn(n, r, dtype=torch.float64, generator=gen)
    w = torch.rand(n, dtype=torch.float64, generator=gen) + 0.05
    I = qdeim_select(torch.linalg.qr(V)[0])
    P = torch.zeros(n, n, dtype=torch.float64)
    P[:, I] = V @ torch.linalg.inv(V[I])
    s = w.sqrt()
    ref = torch.linalg.matrix_norm(s[:, None] * P / s[None, :], 2).item()
    value = lebesgue_constant(V, I, w)
    assert value == pytest.approx(ref, rel=1e-10) and value >= 1.0
    # uniform weights: the unweighted constant
    assert lebesgue_constant(V, I, torch.full((n,), 0.3, dtype=torch.float64)) == (
        pytest.approx(lebesgue_constant(V, I), rel=1e-12)
    )


def _dependent_columns():
    """V = [e0, e1, e0 + e1] (5 x 3): its third column is in the span of the first
    two."""
    V = torch.zeros(5, 3, dtype=torch.float64)
    V[0, 0] = V[1, 1] = V[0, 2] = V[1, 2] = 1.0
    return V


def test_deim_rejects_rank_deficient_basis():
    """The dependent column's interpolation residual vanishes off the rows chosen
    so far: rejected, instead of a chosen row picked again (V(I, :) singular)."""
    with pytest.raises(RuntimeError, match="rank deficient"):
        deim_select(_dependent_columns())


def test_deim_rows_are_distinct():
    """Every selection is r distinct rows, also when the residual is flat."""
    V = torch.linalg.qr(torch.ones(6, 3, dtype=torch.float64) + torch.eye(6, 3))[0]
    rows = deim_select(V)
    assert len(rows) == 3 and len(set(rows)) == 3


def test_lebesgue_constant_of_rank_deficient_basis_is_infinite():
    """A rank-deficient V has no interpolation at r rows: unbounded, whatever I."""
    V = _dependent_columns()
    assert lebesgue_constant(V, [0, 1, 2]) == math.inf
    assert lebesgue_constant(V, qdeim_select(V)) == math.inf


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: deim_select(torch.zeros(3, 5, dtype=torch.float64)), "1 <= r <= n"),
        (
            lambda: deim_select(torch.zeros(5, 2, dtype=torch.float64)),
            "rank deficient",
        ),
        (
            lambda: weighted_orthonormal(
                torch.ones(2, 3, dtype=torch.float64), torch.ones(2)
            ),
            "r <= n",
        ),
        (
            lambda: lebesgue_constant(torch.eye(4, 2, dtype=torch.float64), [0, 1, 2]),
            "I hold r rows",
        ),
        (lambda: qdeim_select(torch.zeros(4, dtype=torch.float64)), "1 <= r <= n"),
        (
            lambda: weighted_orthonormal(
                torch.ones(3, 2, dtype=torch.float64), torch.tensor([1.0, 0.0, 1.0])
            ),
            "positive",
        ),
        (
            lambda: lebesgue_constant(
                torch.eye(3, 2, dtype=torch.float64), [0, 1], torch.ones(2)
            ),
            "w must be",
        ),
    ],
)
def test_rejects_bad_input(call, match):
    with pytest.raises(RuntimeError, match=match):
        call()
