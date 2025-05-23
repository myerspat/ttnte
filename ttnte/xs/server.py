import numpy as np


class Server:
    """
    Cross section (XS) server class. This class handles all XS information for the
    materials in a given problem.

    Attributes
    ----------
    chi: numpy.ndarray
        Fission spectrum of shape ``(Server.num_groups,)``.
    num_groups: int
        Number of energy groups.
    num_moments: int
        Number of scattering moments.
    materials: list of str
        List of material names.
    """

    def __init__(self, xs: dict):
        """
        Initialize XS server class.

        Parameters
        ----------
        xs: dict
            Dictionary with ``"chi"`` spectrum and material XSs. The material XSs
            include the name of the material as the key and a dictionary as the
            value. The dictionary has ``"total"``, ``"nu_fission"``, and
            ``"scattering_gtg"`` arrays.
        """
        self._xs = xs
        self._chi = np.array(self._xs.pop("chi"))

        # Assert chi is 1D and get number of groups
        assert self._chi.ndim == 1
        self._num_groups = self._chi.size

        # Check shapes of groups and determine number of scattering moments
        self._num_moments = None
        for mat in self._xs.keys():
            # Check all arrays are numpy arrays
            self._xs[mat]["total"] = np.array(self.total(mat))
            self._xs[mat]["nu_fission"] = np.array(self.nu_fission(mat))
            self._xs[mat]["scatter_gtg"] = np.array(self.scatter_gtg(mat))

            assert self.total(mat).shape == (self._num_groups,)
            assert self.nu_fission(mat).shape == (self._num_groups,)
            assert self.scatter_gtg(mat).ndim == 3
            assert self.scatter_gtg(mat).shape[1:] == (
                self._num_groups,
                self._num_groups,
            )
            if "absorption" in self._xs[mat]:
                self._xs[mat]["absorption"] = np.array(self.absorption(mat))
                assert self.absorption(mat).shape == (self._num_groups,)
            if "kappa_fission" in self._xs[mat]:
                self._xs[mat]["kappa_fission"] = np.array(self.kappa_fission(mat))
                assert self.kappa_fission(mat).shape == (self._num_groups,)
            if "fission" in self._xs[mat]:
                self._xs[mat]["fission"] = np.array(self.fission(mat))
                assert self.fission(mat).shape == (self._num_groups,)

            if self._num_moments is None:
                self._num_moments = self.scatter_gtg(mat).shape[0]
            else:
                assert self._num_moments == self.scatter_gtg(mat).shape[0]

    # ========================================================================
    # XS accessors

    def total(self, mat):
        """
        Get total XSs.

        Parameters
        ----------
        mat: str
            Material to retrieve XSs for.

        Returns
        -------
        total: numpy.ndarray
            Total XS array of shape ``(Server.num_groups,)``.
        """
        return self._xs[mat]["total"]

    def absorption(self, mat):
        """
        Get absorption XSs.

        Parameters
        ----------
        mat: str
            Material to retrieve XSs for.

        Returns
        -------
        absorption: numpy.ndarray
            Absorption XS array of shape ``(Server.num_groups,)``.
        """
        if "absorption" in self._xs[mat]:
            return self._xs[mat]["absorption"]
        else:
            raise RuntimeError("absorption XSs not provided")

    def nu_fission(self, mat=None):
        """
        Get :math:`\\nu\\Sigma_f` XSs.

        Parameters
        ----------
        mat: str
            Material to retrieve XSs for.

        Returns
        -------
        nu_fission: numpy.ndarray
            :math:`\\nu\\Sigma_f` XS array of shape ``(Server.num_groups,)``.
        """
        return self._xs[mat]["nu_fission"]

    def fission(self, mat=None):
        """
        Get :math:`\\nu\\Sigma_f` XSs.

        Parameters
        ----------
        mat: str
            Material to retrieve XSs for.

        Returns
        -------
        nu_fission: numpy.ndarray
            :math:`\\nu\\Sigma_f` XS array of shape ``(Server.num_groups,)``.
        """
        if "fission" in self._xs[mat]:
            self._xs[mat]["nu_fission"]
        else:
            raise RuntimeError("fission XSs not provided")

    def kappa_fission(self, mat):
        """
        Get :math:`\\kappa\\Sigma_f` XSs.

        Parameters
        ----------
        mat: str
            Material to retrieve XSs for.

        Returns
        -------
        kappa_fission: numpy.ndarray
            :math:`\\kappa\\Sigma_f` XS array of shape ``(Server.num_groups,)``.
        """
        if "kappa_fission" in self._xs[mat]:
            return self._xs[mat]["kappa_fission"]
        else:
            raise RuntimeError("kappa_fission XSs not provided")

    def scatter_gtg(self, mat=None):
        """
        Get group-to-group scattering XSs.

        Parameters
        ----------
        mat: str
            Material to retrieve XSs for.

        Returns
        -------
        total: numpy.ndarray
            Group-to-group scattering XS array of shape
            ``(Server.num_moments, Server.num_groups, Server.num_groups)``.
        """
        return self._xs[mat]["scatter_gtg"]

    # ========================================================================
    # Properties

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
    def materials(self):
        return list(self._xs.keys())
