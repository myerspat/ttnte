"""Matrix-free sum-factorized operator and the ITERATIVE local solver on curved
(rational) patches in 2-D and 3-D."""

import pytest
import torch

from ttnte.sweep import LocalSolverType, SweepConfig

from .problems import BUILDERS, boxes, driver, quadrature

G = 3


def _solved(num_dim, solver, mesh_xs, tol=1e-9, **kw):
    # Local GMRES solves to 1e-11, so the swept operator is exact to about
    # that level and the outer tolerance stays above it
    mesh, xs = mesh_xs
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(tol=tol, max_iter=500, verbose=False, local_solver=solver, **kw)
    sw.setup(quadrature(num_dim, 3), cfg)
    sw.solve_fixed_source()
    assert sw.converged
    return sw


@pytest.mark.parametrize("num_dim", [1, 2, 3])
def test_matrix_free_operator_matches_assembled(num_dim):
    sw = _solved(num_dim, LocalSolverType.BANDED_LU, BUILDERS[num_dim](G))
    torch.manual_seed(0)
    for gid in sw.local_gids:
        disc = sw.get_discretization(gid)
        for omega in sw.ordinates[:5]:
            L = disc.local_operator(omega, 0.7)
            u = torch.randn(disc.num_dofs, dtype=torch.float64)
            y = disc.apply_matrix_free(omega, 0.7, u)
            assert torch.allclose(y, L @ u, rtol=0, atol=1e-12 * (L @ u).abs().max())


@pytest.mark.parametrize("num_dim", [2, 3])
@pytest.mark.parametrize("reflective", [False, True])
def test_iterative_matches_banded_lu_on_curved_patches(num_dim, reflective):
    lu = _solved(num_dim, LocalSolverType.BANDED_LU, BUILDERS[num_dim](G, reflective))
    it = _solved(num_dim, LocalSolverType.ITERATIVE, BUILDERS[num_dim](G, reflective))
    a = torch.cat([it.get_moments(g).flatten() for g in it.local_gids])
    b = torch.cat([lu.get_moments(g).flatten() for g in lu.local_gids])
    assert ((a - b).norm() / b.norm()).item() < 1e-7
    solves, applies, failures = it.iterative_statistics
    assert solves > 0 and failures == 0 and applies / solves < 150


def test_iterative_uses_exact_solver_on_affine_patches():
    it = _solved(2, LocalSolverType.ITERATIVE, boxes(2, G))
    assert it.num_kronecker_units == len(it.local_gids)
    assert it.iterative_statistics == (0, 0, 0)


def test_unconverged_local_solves_fall_back_to_lu():
    lu = _solved(3, LocalSolverType.BANDED_LU, BUILDERS[3](G))
    it = _solved(3, LocalSolverType.ITERATIVE, BUILDERS[3](G), local_max_iter=5)
    assert it.iterative_statistics[2] > 0
    a = torch.cat([it.get_moments(g).flatten() for g in it.local_gids])
    b = torch.cat([lu.get_moments(g).flatten() for g in lu.local_gids])
    assert ((a - b).norm() / b.norm()).item() < 1e-7
