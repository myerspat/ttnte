import numpy as np
import pytest

from tt_nte.methods.discrete_ordinates import DiscreteOrdinates


def test_compute_square_set():
    quadratures = [[8, 32, 128], [4, 16, 64], [8, 32, 128]]

    for i in range(len(quadratures)):
        num_dim = i + 1

        for N in quadratures[i]:
            octant_ords = DiscreteOrdinates._compute_square_set(N, num_dim)
            num_octants = int(2**num_dim)

            # Assertions
            assert octant_ords.shape == (N / num_octants, 1 + num_dim)
            assert np.sum(octant_ords[:, 0]) * num_octants == pytest.approx(1, 1e-10)
            np.testing.assert_array_less(octant_ords, np.ones(octant_ords.shape))
