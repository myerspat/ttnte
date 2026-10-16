"""The Kronecker-sum (SYLVESTER) local solver: a backward-stable complex Schur
decomposition, and sweeps identical to banded LU on axis-aligned patches, with a
fallback to LU on curved ones."""

import pytest
import torch

from ttnte.linalg import complex_schur
from ttnte.sweep import LocalSolverType, SweepConfig

from .problems import BUILDERS, boxes, driver, quadrature

G = 3


def test_complex_schur_non_normal():
    torch.manual_seed(0)
    # A Jordan-like upwind block plus a random dense matrix
    n = 12
    J = torch.eye(n) * 2.0 + torch.diag(torch.full((n - 1,), -1.0), -1)
    for A in (J, torch.randn(n, n), J + 1e-3 * torch.randn(n, n)):
        Q, T = complex_schur(A)
        Ac = A.to(torch.complex128)
        assert torch.allclose(Q @ T @ Q.conj().T, Ac, atol=1e-12 * A.norm())
        assert torch.allclose(
            Q.conj().T @ Q, torch.eye(n, dtype=torch.complex128), atol=1e-12
        )
        assert torch.all(torch.tril(T, -1) == 0)


def _solve(mesh, xs, num_dim, solver, **kw):
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(tol=1e-12, max_iter=500, verbose=False, local_solver=solver, **kw)
    sw.setup(quadrature(num_dim, 3), cfg)
    sw.solve_fixed_source()
    assert sw.converged
    return sw


def _flux(sw):
    return torch.cat([sw.get_moments(g).flatten() for g in sw.local_gids])


@pytest.mark.parametrize("num_dim", [1, 2, 3])
@pytest.mark.parametrize("reflective", [False, True])
def test_sylvester_matches_banded_lu(num_dim, reflective):
    lu = _solve(*boxes(num_dim, G, reflective), num_dim, LocalSolverType.BANDED_LU)
    sy = _solve(*boxes(num_dim, G, reflective), num_dim, LocalSolverType.SYLVESTER)
    assert sy.num_kronecker_units == len(sy.local_gids)
    a, b = _flux(sy), _flux(lu)
    assert ((a - b).norm() / b.norm()).item() < 1e-10


def test_curved_patches_fall_back_to_lu():
    lu = _solve(*BUILDERS[2](G, True), 2, LocalSolverType.BANDED_LU)
    sy = _solve(*BUILDERS[2](G, True), 2, LocalSolverType.SYLVESTER)
    assert sy.num_kronecker_units < len(sy.local_gids)
    a, b = _flux(sy), _flux(lu)
    assert ((a - b).norm() / b.norm()).item() < 1e-10


def test_fibers_with_sylvester():
    sw = _solve(
        *boxes(2, G, True), 2, LocalSolverType.SYLVESTER, store_angular_flux=True
    )
    nd = sw.ordinates.shape[0]
    fibers = sw.sweep_fibers([(list(range(nd)), list(range(G)))]).reshape(nd, G, -1)
    psi = torch.cat(
        [
            sw.get_angular_flux(g).reshape(nd, -1, G).permute(0, 2, 1)
            for g in sw.local_gids
        ],
        2,
    )
    assert ((fibers - psi).norm() / psi.norm()).item() < 1e-9
