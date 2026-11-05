"""Lethargy weighting from the cross-section server's group boundaries: the driver
derives one width per group of the problem and gives each group set's cross its own, so
sets of any size solve to the reference.

The server rejects boundaries that would make a width undefined.
"""

import math

import pytest
import torch

from ttnte.cross import CrossConfig
from ttnte.sweep import SweepConfig, TransportSolve

from .problems import BUILDERS, driver, quadrature

G = 4
WIDTHS = [0.5, 1.0, 2.0, 4.0]


def _bounds(widths, top=2.0e7):
    """Descending boundaries with ln(E_{g-1} / E_g) = widths[g]."""
    bounds = [top]
    for du in widths:
        bounds.append(bounds[-1] * math.exp(-du))
    return bounds


def _phi(num_dim, set_size, transport, bounds=None, tol=1e-6):
    mesh, xs = BUILDERS[num_dim](G)
    if bounds is not None:
        xs.group_bounds = bounds
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=tol,
        max_iter=5000,
        verbose=False,
        group_set_size=set_size,
        transport=transport,
        cross=CrossConfig(tol=1e-6),
    )
    sw.setup(quadrature(num_dim), cfg)
    sw.solve_fixed_source()
    return torch.cat([sw.get_scalar_flux(g).flatten() for g in sw.local_gids])


@pytest.mark.parametrize("num_dim", [1, 2])
@pytest.mark.parametrize("set_size", [1, 3, 4])  # equal sets, unequal (3 + 1), one set
def test_lethargy_widths_per_group_set(num_dim, set_size):
    ref = _phi(num_dim, G, TransportSolve.EXACT, tol=1e-12)
    phi = _phi(num_dim, set_size, TransportSolve.CROSS, _bounds(WIDTHS))
    assert ((phi - ref).norm() / ref.norm()).item() < 1e-4


def test_server_derives_widths():
    _, xs = BUILDERS[1](G)
    assert xs.lethargy_widths == []
    xs.group_bounds = _bounds(WIDTHS)
    assert xs.lethargy_widths == pytest.approx(WIDTHS, rel=1e-12)
    xs.group_bounds = []
    assert xs.lethargy_widths == []


@pytest.mark.parametrize(
    "bounds, match",
    [
        ([2.0e7, 1.0e6, 1.0e3, 1.0, 0.0], "positive"),  # zero lowest edge
        ([2.0e7, 1.0e6, -1.0, 1.0e-3, 1.0e-5], "positive"),
        ([2.0e7, 1.0e6, 1.0e6, 1.0, 1.0e-5], "decreasing"),
        ([1.0e-5, 1.0, 1.0e3, 1.0e6, 2.0e7], "decreasing"),  # ascending
        ([2.0e7, 1.0e6, 1.0], "Expected 5"),  # wrong length for G = 4
    ],
)
def test_server_rejects_bad_bounds(bounds, match):
    _, xs = BUILDERS[1](G)
    with pytest.raises(RuntimeError, match=match):
        xs.group_bounds = bounds
