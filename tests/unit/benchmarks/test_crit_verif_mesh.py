import numpy as np

import tt_nte.benchmarks as benchmarks


def test_pu_brick():
    _, geometry = benchmarks.pu_brick(64)

    # Assertions
    assert geometry.num_nodes == 64
    assert geometry.dx.shape == (63, 1)
    np.testing.assert_array_equal(
        np.round(geometry.dx, 7),
        np.round(3.707444 / 63 * np.ones((63, 1)), 7),
    )
    assert geometry.bcs == ["vacuum", None, None, "vacuum", None, None]
    # np.testing.assert_array_equal(
    #     geometry.region_mask("fuel")[0].flatten(), np.ones(63)
    # )
    assert geometry.num_elements == 63


def test_pu_brick_multi_region():
    _, geometry = benchmarks.pu_brick_multi_region(128, 3)

    # Assertions
    assert geometry.num_nodes == 128
    assert geometry.dx.shape == (127, 1)
    assert geometry.bcs == ["vacuum", None, None, "vacuum", None, None]
    assert geometry.num_elements == 127


def test_research_reactor_multi_region():
    _, geometry = benchmarks.research_reactor_multi_region(
        [33, 32, 33], right_bc="vacuum"
    )

    # Assertions
    assert geometry.num_nodes == 96
    assert geometry.dx.shape == (95, 1)
    # assert np.sum(geometry.region_mask("fuel")[0]) == 31
    # assert np.sum(geometry.region_mask("moderator")[0]) == 64
    assert geometry.num_elements == 95


def test_research_reactor_anisotropic():
    _, geometry = benchmarks.research_reactor_anisotropic(64)

    # Assertions
    assert geometry.num_nodes == 64
    assert geometry.dx.shape == (63, 1)
    np.testing.assert_array_equal(
        np.round(geometry.dx, 7), np.round(9.4959 / 63 * np.ones((63, 1)), 7)
    )
    assert geometry.num_elements == 63


def test_research_reactor_multi_region_2d():
    _, geometry = benchmarks.research_reactor_multi_region_2d(
        [33, 32, 33], 32, right_bc="vacuum"
    )

    # Assertions
    assert geometry.num_nodes == 3072
    assert geometry.dx.shape == (95, 1)
    assert geometry.dy.shape == (31, 1)
    assert geometry.bcs == ["vacuum", "reflective", None, "vacuum", "reflective", None]
    # assert np.sum(geometry.region_mask("fuel")[0]) == 31
    # assert np.sum(geometry.region_mask("fuel")[1]) == 31
    # assert np.sum(geometry.region_mask("moderator")[0]) == 64
    # assert np.sum(geometry.region_mask("moderator")[1]) == 31
    assert geometry.num_elements == 2945
