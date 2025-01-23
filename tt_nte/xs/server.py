class Server:
    def __init__(self, xs):
        self._xs = xs
        self._chi = self._xs.pop("chi")

        # Assert chi is 1D and get number of groups
        assert len(self._chi.shape) == 1
        self._num_groups = self._chi.size

        # Check if the XSs are given for each cell or for material type
        if "total" in self._xs:
            self._by_mat = False
            self._num_moments = self._xs["scatter_gtg"].shape[0]

            # Check shapes
            assert self._xs["total"].shape[0] == self._num_groups
            assert self._xs["nu_fission"].shape[0] == self._num_groups
            assert self._xs["scatter_gtg"].shape[1:3] == (
                self._num_groups,
                self._num_groups,
            )
            assert self._xs["total"].shape[1:] == self._xs["nu_fission"].shape[1:]
            assert self._xs["total"].shape[1:] == self._xs["scatter_gtg"].shape[3:]

        else:
            self._by_mat = True
            self._num_moments = None

            for mat in self._xs.keys():
                assert self.total(mat).shape == (self._num_groups,)
                assert self.nu_fission(mat).shape == (self._num_groups,)
                assert len(self.scatter_gtg(mat).shape) == 3
                assert self.scatter_gtg(mat).shape[1:] == (
                    self._num_groups,
                    self._num_groups,
                )

                if self._num_moments is None:
                    self._num_moments = self.scatter_gtg(mat).shape[0]
                else:
                    assert self._num_moments == self.scatter_gtg(mat).shape[0]

    def total(self, mat=None):
        if self._by_mat:
            return self._xs[mat]["total"]
        else:
            return self._xs["total"]

    def nu_fission(self, mat=None):
        if self._by_mat:
            return self._xs[mat]["nu_fission"]
        else:
            return self._xs["nu_fission"]

    def scatter_gtg(self, mat=None):
        if self._by_mat:
            return self._xs[mat]["scatter_gtg"]
        else:
            return self._xs["scatter_gtg"]

    @property
    def chi(self):
        return self._chi

    @property
    def num_groups(self):
        return self._num_groups

    @property
    def num_moments(self):
        return self._num_moments

    @property
    def by_mat(self):
        return self._by_mat
