"""The SweepConfig knobs shared by the sampled transport solves (CROSS, GALERKIN) are
sampling_*; the old cross_* names are deprecated aliases (attributes and keyword
arguments) that warn and forward."""

import pytest

from ttnte.sweep import SweepConfig

ALIASES = [
    ("cross_inner_factor", "sampling_inner_factor", 0.25),
    ("cross_relax", "sampling_relax", False),
    ("cross_max_tol", "sampling_max_tol", 3e-3),
    ("cross_threads", "sampling_threads", 3),
]


@pytest.mark.parametrize("old, new, value", ALIASES)
def test_alias_forwards_and_warns(old, new, value):
    cfg = SweepConfig()
    with pytest.warns(FutureWarning, match=new):
        setattr(cfg, old, value)
    assert getattr(cfg, new) == value
    with pytest.warns(FutureWarning, match=new):
        assert getattr(cfg, old) == value


@pytest.mark.parametrize("old, new, value", ALIASES)
def test_alias_keyword(old, new, value):
    with pytest.warns(FutureWarning, match=new):
        cfg = SweepConfig(**{old: value})
    assert getattr(cfg, new) == value


def test_defaults_unchanged():
    cfg = SweepConfig()
    assert cfg.sampling_inner_factor == 0.1 and cfg.sampling_relax
    assert cfg.sampling_max_tol == 1e-2 and cfg.sampling_threads == 16
