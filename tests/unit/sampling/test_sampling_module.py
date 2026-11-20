"""The sampling module holds what the sweep-sampled transport solves (cross, Galerkin)
share; Weighting stays importable from ttnte.cross."""

import ttnte.cross
import ttnte.sampling
from ttnte.sampling import Weighting


def test_weighting_lives_in_sampling_and_is_aliased_in_cross():
    assert ttnte.cross.Weighting is ttnte.sampling.Weighting
    w = Weighting(angle=False)
    assert not w.angle and w.energy and w.space
    assert isinstance(ttnte.cross.CrossConfig().weighting, Weighting)


def test_record_classes():
    """CrossRun and GalerkinRun are RunRecords (pybind returns the subclass)."""
    from ttnte.cross import CrossRun
    from ttnte.galerkin import GalerkinRun
    from ttnte.sampling import RunRecord

    assert issubclass(CrossRun, RunRecord) and issubclass(GalerkinRun, RunRecord)
