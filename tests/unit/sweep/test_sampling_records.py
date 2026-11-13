"""SweepDriver.sampling_records: one record per sampled transport solve, of the
sampler's own class; the old cross_* records are deprecated aliases computed from
them."""

import pytest

from ttnte.cross import CrossRun
from ttnte.galerkin import GalerkinRun
from ttnte.sampling import RunRecord
from ttnte.sweep import SweepConfig, TransportSolve

from .problems import BUILDERS, driver, quadrature

G = 3


def _solve(transport):
    mesh, xs = BUILDERS[1](G, True)
    sw = driver(1, mesh, xs)
    sw.setup(
        quadrature(1),
        SweepConfig(tol=1e-6, max_iter=200, verbose=False, transport=transport),
    )
    sw.solve_fixed_source()
    return sw


@pytest.mark.parametrize(
    "transport, cls",
    [(TransportSolve.CROSS, CrossRun), (TransportSolve.GALERKIN, GalerkinRun)],
)
def test_records_and_deprecated_aliases(transport, cls):
    sw = _solve(transport)
    records = sw.sampling_records
    assert len(records) > 0
    assert all(isinstance(r, cls) and isinstance(r, RunRecord) for r in records)
    assert all(len(r.history) > 0 for r in records)
    assert sw.sampling_seconds >= sw.sampling_sweep_seconds > 0.0
    assert "moments" in sw.sampling_timings
    expected = {
        "cross_fibers": [r.fibers for r in records],
        "cross_tolerances": [r.tolerance for r in records],
        "cross_errors": [r.error for r in records],
        "cross_ranks": [list(r.ranks) for r in records],
        "cross_max_ranks": [r.max_rank for r in records],
        "cross_seconds": sw.sampling_seconds,
        "cross_sweep_seconds": sw.sampling_sweep_seconds,
        "cross_timings": sw.sampling_timings,
    }
    for old, value in expected.items():
        with pytest.warns(FutureWarning, match="sampling"):
            assert getattr(sw, old) == value
    with pytest.warns(FutureWarning):
        histories = sw.cross_histories
    assert len(histories) == (len(records) if cls is CrossRun else 0)


def test_exact_transport_has_no_records():
    sw = _solve(TransportSolve.EXACT)
    assert sw.sampling_records == [] and sw.sampling_seconds == 0.0
