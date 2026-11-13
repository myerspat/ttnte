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
