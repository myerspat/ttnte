"""The SweepConfig knobs shared by the sampled transport solves (CROSS, GALERKIN) are
sampling_*; the old cross_* aliases are gone (attributes and keyword arguments)."""

import pytest

from ttnte.sweep import SweepConfig

REMOVED = ["cross_inner_factor", "cross_relax", "cross_max_tol", "cross_threads"]


@pytest.mark.parametrize("old", REMOVED)
def test_old_names_removed(old):
    cfg = SweepConfig()
    with pytest.raises(AttributeError):
        getattr(cfg, old)
    with pytest.raises((TypeError, AttributeError, ValueError, RuntimeError)):
        SweepConfig(**{old: 1})


def test_defaults_unchanged():
    cfg = SweepConfig()
    assert cfg.sampling_inner_factor == 0.1 and cfg.sampling_relax
    assert cfg.sampling_max_tol == 1e-2 and cfg.sampling_threads == 16
