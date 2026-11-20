"""A new solve restarts the samplers: solving again on the same driver gives what a
freshly set-up driver gives (same flux, same sampled runs), for the cross, the greedy
Galerkin, and the Galerkin's DEIM stage (whose stage, references and DEIM rows used to
carry over into the next solve)."""

import pytest
import torch

from ttnte.galerkin import DeimConfig, GalerkinConfig
from ttnte.sweep import InnerSolver, SweepConfig, TransportSolve

from .problems import BUILDERS, driver, quadrature

G = 3

VARIANTS = {
    "cross": dict(transport=TransportSolve.CROSS),
    "galerkin": dict(transport=TransportSolve.GALERKIN),
    "galerkin-deim": dict(
        transport=TransportSolve.GALERKIN,
        inner_solver=InnerSolver.RICHARDSON,
        galerkin=GalerkinConfig(deim=DeimConfig(enabled=True)),
    ),
}


def _driver(variant):
    mesh, xs = BUILDERS[1](G, True)
    sw = driver(1, mesh, xs)
    sw.setup(
        quadrature(1, 16),
        SweepConfig(tol=1e-6, max_iter=500, verbose=False, **VARIANTS[variant]),
    )
    return sw


def _solve(sw):
    sw.solve_fixed_source()
    assert sw.converged
    phi = torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids])
    return phi, [(type(r).__name__, r.fibers) for r in sw.sampling_records]


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_second_solve_equals_fresh_driver(variant):
    sw = _driver(variant)
    first = _solve(sw)
    second = _solve(sw)
    fresh = _solve(_driver(variant))
    assert second[1] == fresh[1]
    assert torch.equal(second[0], fresh[0])
    assert torch.equal(first[0], fresh[0])
