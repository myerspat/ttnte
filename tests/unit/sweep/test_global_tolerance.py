"""SweepConfig.global_inner_tolerance: within-set tolerances relative to the whole
scalar flux.

With one group per set the sets hold different shares of the flux; the solve
must still converge to the reference within the outer tolerance, and it may not
need more group-set sweeps.
"""

import pytest
import torch

from ttnte.cross import CrossConfig
from ttnte.sweep import SweepConfig, TransportSolve

from .problems import BUILDERS, driver, quadrature

G = 3


def _solve(num_dim, transport, global_tol, tol=1e-5):
    mesh, xs = BUILDERS[num_dim](G)
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=tol,
        max_iter=5000,
        verbose=False,
        group_set_size=1,
        transport=transport,
        cross=CrossConfig(tol=1e-4),
        global_inner_tolerance=global_tol,
    )
    sw.setup(quadrature(num_dim), cfg)
    sw.solve_fixed_source()
    return sw


def _phi(sw):
    return torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids])


@pytest.mark.parametrize("num_dim", [1, 2])
@pytest.mark.parametrize("transport", [TransportSolve.EXACT, TransportSolve.CROSS])
def test_global_inner_tolerance(num_dim, transport):
    ref = _phi(_solve(num_dim, TransportSolve.EXACT, False, tol=1e-12))
    local = _solve(num_dim, transport, False)
    glob = _solve(num_dim, transport, True)
    for sw in (local, glob):
        assert sw.converged
        assert ((_phi(sw) - ref).norm() / ref.norm()).item() < 1e-3
    assert glob.group_set_sweeps <= local.group_set_sweeps
    assert SweepConfig().global_inner_tolerance is False


@pytest.mark.parametrize("transport", [TransportSolve.EXACT, TransportSolve.CROSS])
def test_global_inner_tolerance_eigenvalue(transport):
    """With fission the factor is capped by the set's share of the fission production; k
    stays within the outer tolerance of the reference."""

    def k_eff(global_tol, tol):
        mesh, xs = BUILDERS[1](G, False, True)
        sw = driver(1, mesh, xs)
        cfg = SweepConfig(
            tol=tol,
            max_iter=5000,
            verbose=False,
            group_set_size=1,
            transport=transport if tol > 1e-10 else TransportSolve.EXACT,
            cross=CrossConfig(tol=1e-4),
            global_inner_tolerance=global_tol,
        )
        sw.setup(quadrature(1), cfg)
        sw.solve_eigenvalue(tol)
        return sw.k_eff

    ref = k_eff(False, 1e-11)
    assert abs(k_eff(True, 1e-5) - ref) < 1e-4
