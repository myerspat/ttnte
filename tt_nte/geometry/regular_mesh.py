import numpy as np

from .geometry import Geometry


class RegularMesh(Geometry):
    def __init__(self, size, num_nodes, bcs):
        self._size = size
        self._num_nodes = np.prod(num_nodes)
        self._num_elements = np.prod([n - 1 for n in num_nodes])
        self._bcs = bcs
        self._num_dim = len(size)

        # Check sizes
        assert self._num_dim == len(num_nodes)
        assert 2 * 3 == len(self._bcs)

        # Assumes start at zero and calculate deltas
        self._diff = 3 * [None]
        for i, size in enumerate(self._size):
            self._diff[i] = size / (num_nodes[i] - 1) * np.ones(num_nodes[i] - 1)

    # ===================================================
    # Getters

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_elements(self):
        return self._num_elements

    @property
    def num_dim(self):
        return self._num_dim

    @property
    def diff(self):
        return self._diff

    @property
    def dx(self):
        return self._diff[0]

    @property
    def dy(self):
        return self._diff[1]

    @property
    def dz(self):
        return self._diff[2]

    @property
    def bcs(self):
        return self._bcs
