"""SweepConfig.certify: after a CROSS solve, one exact transport solve of every group
set at the converged sources replaces the cross's moments and reports the scalar-flux
change (and the change in k)."""

import math

import pytest
import torch

from ttnte.cross import CrossConfig
from ttnte.sweep import SweepConfig, TransportSolve

from .problems import BUILDERS, driver, quadrature

G = 3


def _solve(num_dim, reflective, transport, certify, fissile=False, tol=1e-4):
    mesh, xs = BUILDERS[num_dim](G, reflective, fissile)
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=tol,
        max_iter=5000,
        verbose=False,
        transport=transport,
        cross=CrossConfig(tol=1e-4),
        certify=certify,
    )
    sw.setup(quadrature(num_dim), cfg)
    if fissile:
        sw.solve_eigenvalue(tol)
    else:
        sw.solve_fixed_source()
    return sw


def _phi(sw):
    return torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids])


@pytest.mark.parametrize(
    "num_dim, reflective", [(1, False), (1, True), (2, False), (2, True)]
)
def test_certified_fixed_source(num_dim, reflective):
    ref = _phi(_solve(num_dim, reflective, TransportSolve.EXACT, False, tol=1e-12))
    plain = _solve(num_dim, reflective, TransportSolve.CROSS, False)
    assert math.isnan(plain.certification_error)
    cert = _solve(num_dim, reflective, TransportSolve.CROSS, True)
    # The certificate is the change the exact solve makes to the cross's flux
    before = _phi(plain)
    after = _phi(cert)
    change = ((after - before).norm() / after.norm()).item()
    assert cert.certification_error == pytest.approx(change, rel=1e-6, abs=1e-14)
    assert 0.0 < cert.certification_error < 1e-2
    # One exact transport solve from a near-converged flux does not hurt
    err_before = ((before - ref).norm() / ref.norm()).item()
    err_after = ((after - ref).norm() / ref.norm()).item()
    assert err_after <= 1.5 * err_before + 1e-10
    assert cert.certification_seconds > 0.0


def test_certified_eigenvalue():
    ref = _solve(1, True, TransportSolve.EXACT, False, fissile=True, tol=1e-12)
    cert = _solve(1, True, TransportSolve.CROSS, True, fissile=True)
    assert 0.0 <= cert.certification_k_change < 1e-3
    assert abs(cert.k_eff - ref.k_eff) < 1e-3


def test_exact_transport_is_not_certified():
    sw = _solve(1, False, TransportSolve.EXACT, True)
    assert math.isnan(sw.certification_error)
