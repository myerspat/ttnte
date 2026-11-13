"""linalg.mgs2a: Gram-Schmidt with reorthogonalization and adaptive cleanups (MGS2-A,
Guo & Peng Alg.

3.2) of one vector against an orthonormal basis.
"""

import pytest
import torch

from ttnte.linalg import MGS2Options, mgs2a

N = 50


def _basis(r, n=N, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(n, r, generator=g, dtype=dtype))[0]
    return Q.t().contiguous()


def _random(n=N, dtype=torch.float64, seed=1):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=dtype)


@pytest.mark.parametrize("r", [0, 1, 7, 30])
def test_reproduces_and_orthogonalizes(r):
    """X = Q^T c + alpha u, u a unit vector orthogonal to Q to rounding."""
    Q = _basis(r)
    x0 = _random()
    x = x0.clone()
    res = mgs2a(Q, x)
    assert res.appended and res.orthogonal
    assert res.coefficients.shape == (r,)
    assert res.norm == pytest.approx(x0.norm().item(), rel=1e-15)
    assert abs(x.norm().item() - 1.0) < 1e-14
    assert (Q @ x).norm().item() < 1e-14
    rebuilt = Q.t() @ res.coefficients + res.alpha * x
    assert (rebuilt - x0).norm() / x0.norm() < 1e-14


def test_nearly_dependent_input_is_orthogonalized():
    """A vector in the span up to 1e-9: one projection would leave about
    eps / 1e-9 = 1e-7 of it along Q; two leave rounding, without cleanups."""
    Q = _basis(10)
    x0 = Q.t() @ _random(10) + 1e-9 * _random()
    x = x0.clone()
    res = mgs2a(Q, x, MGS2Options(max_projections=2))
    assert res.appended and res.orthogonal
    assert (Q @ x).abs().max().item() < 1e-13
    assert res.alpha / res.norm < 1e-8


@pytest.mark.parametrize("rtol, appended", [(0.0, True), (1e-6, False)])
def test_rtol_drops_snapshots_in_the_span(rtol, appended):
    """Remainder 1e-9 ||x||: appended at rtol = 0 (the paper), not at 1e-6, where the
    coefficients alone reproduce x to the tolerance."""
    Q = _basis(10)
    x0 = Q.t() @ _random(10) + 1e-9 * _random()
    x = x0.clone()
    res = mgs2a(Q, x, MGS2Options(rtol=rtol))
    assert res.appended == appended
    if not appended:
        assert (Q.t() @ res.coefficients - x0).norm() / x0.norm() < 1e-8


def test_full_basis_and_zero_vector_append_nothing():
    Q = _basis(N)
    x0 = _random()
    res = mgs2a(Q, x0.clone())
    assert not res.appended
    assert (Q.t() @ res.coefficients - x0).norm() / x0.norm() < 1e-13
    res = mgs2a(_basis(5), torch.zeros(N, dtype=torch.float64))
    assert not res.appended and res.norm == 0.0 and res.alpha == 0.0


def test_cleanups_repair_a_lossy_basis():
    """Q whose rows are orthonormal only to 1e-2: each projection reduces the
    component along q_1 by about that factor, so two (max_projections = 2) leave
    about 1e-4 (an orthogonality failure at eps = 1e-10), while the default
    cleanups (6 projections in all) bring it below eps."""
    Q = _basis(10)
    Q[1] += 1e-2 * Q[0]
    x = _random()
    res = mgs2a(Q, x.clone(), MGS2Options(max_projections=2))
    assert res.appended and not res.orthogonal
    u = x.clone()
    res = mgs2a(Q, u)
    assert res.appended and res.orthogonal
    assert abs(Q[0].dot(u).item()) <= 1e-10


def test_float32():
    Q = _basis(7, dtype=torch.float32)
    x0 = _random(dtype=torch.float32)
    x = x0.clone()
    res = mgs2a(Q, x, MGS2Options(eps=1e-5))
    assert res.appended and res.coefficients.dtype == torch.float32
    assert (Q @ x).abs().max().item() < 1e-6
    rebuilt = Q.t() @ res.coefficients + res.alpha * x
    assert (rebuilt - x0).norm() / x0.norm() < 1e-6


@pytest.mark.parametrize(
    "Q, x, options, match",
    [
        (torch.zeros(3, N), torch.zeros(N + 1), MGS2Options(), "Expected"),
        (
            torch.zeros(3, N),
            torch.zeros(N, dtype=torch.float32),
            MGS2Options(),
            "dtype",
        ),
        (torch.zeros(3, N), torch.zeros(N), MGS2Options(max_projections=1), "max_proj"),
        (torch.zeros(3, N), torch.zeros(N), MGS2Options(eps=0.0), "eps"),
        (torch.zeros(3, N), torch.zeros(N), MGS2Options(rtol=-1.0), "rtol"),
        (torch.zeros(3, N), torch.zeros(2 * N)[::2], MGS2Options(), "contiguous"),
    ],
)
def test_rejects_bad_input(Q, x, options, match):
    with pytest.raises(RuntimeError, match=match):
        mgs2a(Q.double(), x if x.dtype == torch.float32 else x.double(), options)
