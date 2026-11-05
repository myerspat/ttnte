"""SweepConfig.cache_factorizations fills per (direction, group) pair from whichever
sweep factors it first, so a CROSS solve (fiber sweeps only) uses the cache too and
gives the same result as without it."""

import pytest
import torch

from ttnte.cross import CrossConfig
from ttnte.sweep import SweepConfig, TransportSolve

from .problems import BUILDERS, driver, quadrature

G = 3


def _solve(num_dim, transport, cache):
    mesh, xs = BUILDERS[num_dim](G)
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=1e-8,
        max_iter=5000,
        verbose=False,
        transport=transport,
        cache_factorizations=cache,
        num_threads=4,
        cross=CrossConfig(tol=1e-6, seed=3),
    )
    sw.setup(quadrature(num_dim), cfg)
    sw.solve_fixed_source()
    return torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids])


@pytest.mark.parametrize("num_dim", [1, 2])
@pytest.mark.parametrize("transport", [TransportSolve.EXACT, TransportSolve.CROSS])
def test_cache_matches_uncached(num_dim, transport):
    ref = _solve(num_dim, transport, False)
    cached = _solve(num_dim, transport, True)
    assert ((cached - ref).norm() / ref.norm()).item() < 1e-10
