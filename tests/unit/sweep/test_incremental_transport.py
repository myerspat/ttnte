"""SweepConfig.incremental_transport: y0 = y_prev + T(change of the input).

T is linear, so with EXACT transport the result must match the plain correction
form; increments in a row are capped at 1 / cross_inner_factor (each may add
inner_factor of the inner tolerance to y0).
"""

import pytest
import torch

from ttnte.sweep import SweepConfig, TransportSolve

from .problems import BUILDERS, driver, quadrature

G = 3


def _solve(incremental, inner_factor=0.1, num_dim=1):
    mesh, xs = BUILDERS[num_dim](G, False, True)
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=1e-7,
        max_iter=5000,
        verbose=False,
        group_set_size=1,
        transport=TransportSolve.EXACT,
        correction_form=True,
        incremental_transport=incremental,
        cross_inner_factor=inner_factor,
    )
    sw.setup(quadrature(num_dim), cfg)
    sw.solve_eigenvalue(1e-8)
    return sw


def _phi(sw):
    return torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids])


@pytest.mark.parametrize("num_dim", [1, 2])
def test_exact_incremental_matches_plain(num_dim):
    plain = _solve(False, num_dim=num_dim)
    inc = _solve(True, num_dim=num_dim)
    assert inc.incremental_solves > 0
    assert abs(inc.k_eff - plain.k_eff) < 1e-6
    assert ((_phi(inc) - _phi(plain)).norm() / _phi(plain).norm()).item() < 1e-5


@pytest.mark.parametrize("inner_factor", [0.5, 0.25])
def test_increments_in_a_row_are_capped(inner_factor):
    sw = _solve(True, inner_factor)
    per_full = int(round(1.0 / inner_factor))  # drift per_full * f * tol <= tol
    num_sets = G  # one group per set
    assert sw.full_solves >= num_sets
    assert sw.incremental_solves <= per_full * sw.full_solves
