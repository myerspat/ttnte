"""The pivoting and selection helpers of the cross (ttnte/cross/pivots.hpp):

column-pivoted QR for the initial energies, the per-direction Gaussian sketch,
and block / single-row selection of index sets.
"""

import pytest
import scipy.linalg
import torch

from ttnte.cross import (
    direction_sketch,
    pivot_columns,
    select_block_grams,
    select_single_rows,
)

torch.set_default_dtype(torch.float64)


def _low_rank(k, n, r, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, r, generator=g) @ torch.randn(r, n, generator=g)


# =================================================================
# pivot_columns


@pytest.mark.parametrize("r", [1, 3, 6])
def test_pivot_columns_finds_the_rank(r):
    R = _low_rank(40, 25, r)
    assert len(pivot_columns(R, 1e-10)) == r


def test_pivot_columns_matches_reference_qrcp():
    R = _low_rank(40, 25, 8, seed=3)
    picks = pivot_columns(R, 1e-10)
    _, _, ref = scipy.linalg.qr(R.numpy(), pivoting=True)
    assert picks == list(ref[: len(picks)])


@pytest.mark.parametrize("eps", [1e-1, 1e-2, 1e-4])
def test_pivot_columns_meets_its_tolerance(eps):
    """The columns left out are within eps ||R||_F of the span of the picks, and one
    pick fewer would not be."""
    g = torch.Generator().manual_seed(1)
    U, _ = torch.linalg.qr(torch.randn(60, 20, generator=g))
    V, _ = torch.linalg.qr(torch.randn(30, 20, generator=g))
    R = (U * 0.5 ** torch.arange(20.0)) @ V.T
    picks = pivot_columns(R, eps)

    def residual(cols):
        Q, _ = torch.linalg.qr(R[:, cols])
        return torch.linalg.norm(R - Q @ (Q.T @ R)).item()

    total = torch.linalg.norm(R).item()
    assert residual(picks) <= eps * total * (1 + 1e-12)
    if len(picks) > 1:
        assert residual(picks[:-1]) > eps * total


def test_pivot_columns_returns_at_least_one():
    assert pivot_columns(torch.zeros(5, 4), 1e-3) == [0]
    assert len(pivot_columns(torch.ones(5, 4), 1e-3)) == 1


# =================================================================
# direction_sketch


def test_direction_sketch_deterministic_per_direction():
    a = direction_sketch(3, 8, 100)
    assert a.shape == (8, 100)
    assert torch.equal(a, direction_sketch(3, 8, 100))
    assert not torch.equal(a, direction_sketch(4, 8, 100))


def test_direction_sketch_is_standard_normal():
    s = direction_sketch(7, 200, 1000).flatten()
    assert abs(s.mean().item()) < 0.02
    assert abs(s.var().item() - 1.0) < 0.02


# =================================================================
# select_single_rows and select_block_grams


def _cond(rows):
    s = torch.linalg.svdvals(rows)
    return (s[-1] / s[0]).item()


@pytest.mark.parametrize("cond", [1e-2, 0.2])
def test_single_rows_reach_the_conditioning(cond):
    B = torch.randn(80, 6, generator=torch.Generator().manual_seed(2))
    picked = select_single_rows(B, [1.0] * 80, [False] * 80, cond, 0.25)
    assert sum(picked) >= 6
    assert _cond(B[torch.tensor(picked)]) >= cond


def test_single_rows_keep_forced_and_prefer_cheap():
    """Forced rows stay chosen, and of two equal rows the cheaper is taken."""
    B = torch.randn(40, 3, generator=torch.Generator().manual_seed(4))
    B[1] = B[0] * 10.0  # rows 0 and 1 the same direction, both large
    B[2] = B[0] * 10.0
    cost = [1.0] * 40
    cost[1] = 5.0
    cost[2] = 0.0
    forced = [False] * 40
    forced[7] = True
    picked = select_single_rows(B, cost, forced, 1e-2, 0.25)
    assert picked[7]
    assert picked[2] and not picked[1]


def test_block_grams_match_single_rows_on_one_row_blocks():
    B = torch.randn(60, 5, generator=torch.Generator().manual_seed(5))
    grams = torch.stack([B[i : i + 1].T @ B[i : i + 1] for i in range(60)])
    by_gram = select_block_grams(grams, [1] * 60, [1.0] * 60, [False] * 60, 1e-2, 0.25)
    by_row = select_single_rows(B, [1.0] * 60, [False] * 60, 1e-2, 0.25)
    assert by_gram == by_row


def test_block_grams_reach_the_conditioning_with_blocks():
    """Blocks of several rows: the stacked rows of the chosen blocks are
    conditioned to cond, and forced blocks stay chosen."""
    g = torch.Generator().manual_seed(6)
    blocks = [
        torch.randn(int(m), 4, generator=g)
        for m in torch.randint(1, 5, (30,), generator=g)
    ]
    grams = torch.stack([b.T @ b for b in blocks])
    sizes = [b.shape[0] for b in blocks]
    forced = [False] * 30
    forced[11] = True
    picked = select_block_grams(grams, sizes, [1.0] * 30, forced, 0.1, 0.25)
    assert picked[11]
    rows = torch.cat([b for b, p in zip(blocks, picked) if p])
    assert rows.shape[0] >= 4
    assert _cond(rows) >= 0.1


def test_single_rows_keep_conditioning_once_the_span_is_full():
    """Rows that fill the span but leave one direction weak: once the span is
    full, the Gram-Schmidt residuals are zero (only round-off), so further
    rows are scored, as in select_block_grams, by their energy in the poorly
    resolved directions. The few rows that strengthen it are taken and the
    target is met with them, not with rows ranked by round-off."""
    g = torch.Generator().manual_seed(8)
    B = torch.zeros(80, 6)
    B[:60, :5] = torch.randn(60, 5, generator=g)  # strong in five directions
    weak = torch.arange(60, 80)
    B[weak, 5] = 0.02 * (1 + 0.1 * torch.rand(20, generator=g))  # weak sixth
    picked = select_single_rows(B, [1.0] * 80, [False] * 80, 1e-2, 0.25)
    idx = [i for i, p in enumerate(picked) if p]
    assert _cond(B[torch.tensor(idx)]) >= 1e-2
    strong = [i for i in idx if i < 60]
    assert len(strong) <= 6 and len(idx) - len(strong) <= 10, idx
    # The target is unreachable without the weak rows' direction
    assert sum(i >= 60 for i in idx) >= 2
