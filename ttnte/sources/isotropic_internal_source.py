import itertools
from typing import Union

import h5py as h5
import numpy as np
import torch as tn
from geomdl import NURBS as GeomdlNURBS
from igakit.nurbs import NURBS as IgakitNURBS

from ttnte.cad._utils import _igakit2geomdl


class IsotropicInternalSource:
    """
    Isotropic internal source object. This defines an internal source using NURBS. This
    class evaluates.

    .. math::

        Q_{g} = \\sum_a\\sum_bR_{a,b}^{p,q}(\\hat{x}, \\hat{y})\\mathbf{Q}_{a,b}

    for a given :math:`(\\hat{x}, \\hat{y})` for each patch.
    """

    # Source ID iterator
    _id_iter = itertools.count()

    def __init__(self, source_ctrlpts: np.ndarray):
        """Initialize IsotropicInternalSource object."""
        assert source_ctrlpts.ndim == 3
        self._converted = False
        self._pid = None
        self._nurbs = None
        self._source_ctrlpts = source_ctrlpts
        self._id = next(self._id_iter)

    # ========================================================================
    # Public methods

    def set_patch(self, pid: int, nurbs: Union[IgakitNURBS, GeomdlNURBS.Surface]):
        """
        Set the patch of the isotropic internal source

        Parameters
        ----------
        pid: int
            Patch ID corresponding to the ``ttnte.cad.Patch``.
        patch: ttnte.cad.Patch
            Patch object.
        """
        assert (
            self._converted == False and nurbs.shape
            if isinstance(nurbs, IgakitNURBS)
            else (nurbs.ctrlpts_size_u, nurbs.ctrlpts_size_v)
            == self._source_ctrlpts.shape[1:]
        )
        self._pid = pid
        self._nurbs = nurbs
        self._source_ctrlpts = self._source_ctrlpts.reshape(
            (self._source_ctrlpts.shape[0], -1)
        )

    def igakit2geomdl(self):
        """Convert ``igakit.nurbs.NURBS`` to ``geomdl.NURBS.Surface``."""
        assert self._converted == False and isinstance(self._nurbs, IgakitNURBS)
        self._nurbs = _igakit2geomdl(self._nurbs)
        self._converted = True

    def evaluate(self, coords: tn.Tensor, num_ordinates: int):
        """
        Evaluate the internal source.

        Parameters
        ----------
        coords: tn.Tensor
            Parametric coordinates for evaluating the internal source.
            ``coords`` should be an array of shape ``(n, 2)`` where
            ``n`` is the number of coordinates to evaluate. The first
            index in the first dimension correspons to ``xhat`` while
            the second correspons to ``yhat``.

        Returns
        -------
        source_data: numpy.ndarray or None
            Internal source of patch ``pid`` of shape
            ``(ttnte.xs.Server.num_groups, n)``. If ``pid`` is not
            defined as an isotropic internal source then returns ``None``.
        """
        return self.__call__(coords, num_ordinates)

    # ========================================================================
    # Overloads

    def __call__(self, coords: tn.Tensor, num_ordinates: int):
        """
        Evaluate the internal source.

        Parameters
        ----------
        coords: tn.Tensor
            Parametric coordinates for evaluating the internal source.
            ``coords`` should be an array of shape ``(n, 2)`` where
            ``n`` is the number of coordinates to evaluate. The first
            index in the first dimension correspons to ``xhat`` while
            the second correspons to ``yhat``.

        Returns
        -------
        source_data: numpy.ndarray or None
            Internal source of patch ``pid`` of shape
            ``(ttnte.xs.Server.num_groups, n)``. If ``pid`` is not
            defined as an isotropic internal source then returns ``None``.
        """
        assert self._converted and coords.ndim == 2 and coords.shape[1] == 2

        # Array to fill
        results = np.zeros((self._source_ctrlpts.shape[0], coords.shape[0]))

        # Get the control points
        ctrlpts = self._nurbs.ctrlpts

        # Iterate through groups
        for g in range(self._source_ctrlpts.shape[0]):
            # Set the control points
            for i in range(len(ctrlpts)):
                ctrlpts[i] = [*ctrlpts[i][:-1], self._source_ctrlpts[g, i, ...]]
            self._nurbs.ctrlpts = ctrlpts

            # Evaluate coordinates
            results[g, ...] = np.array(self._nurbs.evaluate_list(coords))[:, -1]

        return np.array([results for _ in range(num_ordinates)])

    def __str__(self):
        return "IsotropicInternalSource"

    def __repr__(self):
        return self.__str__()

    # ========================================================================
    # I / O

    def save(self, pg: h5.Group):
        """
        Save internal source information

        Parameters
        ----------
        pg: h5py.Group
            Patch group that the source belongs to.

        Returns
        -------
        pg: h5py.Group
            Patch group that the source belongs to.
        """
        assert self._converted

        # Create group for patch
        sg = pg.create_group(f"Source")
        sg.attrs["Patch ID"] = self._pid

        # Save degree
        dg = sg.create_group("Degree")
        dg.attrs["u"], dg.attrs["v"] = self._nurbs.degree_u, self._nurbs.degree_v

        # Save control points
        cp = sg.create_group("Control Points")
        cp.attrs["u"], cp.attrs["v"] = (
            self._nurbs.ctrlpts_size_u,
            self._nurbs.ctrlpts_size_v,
        )
        cp.create_dataset(
            "u", data=np.array(self._nurbs.ctrlpts).reshape((-1, 3))[:, 0]
        )
        cp.create_dataset(
            "v", data=np.array(self._nurbs.ctrlpts).reshape((-1, 3))[:, 1]
        )
        cp.create_dataset(
            "s", data=self._source_ctrlpts.reshape((self._source_ctrlpts.shape[0], -1))
        )

        # Save weights
        sg.create_dataset("Weights", data=np.array(self._nurbs.weights).flatten())

        # Save knot vectors
        kg = sg.create_group("Knot Vectors")
        kg.create_dataset("u", data=self._nurbs.knotvector_u[0])
        kg.create_dataset("v", data=self._nurbs.knotvector_v[1])

        return pg

    @classmethod
    def load(cls, sg: h5.Group):
        """
        Load internal source.

        Parameters
        ----------
        sg: h5py.Group
            Source group for the patch.

        Returns
        -------
        source: ttnte.source.IsotropicInternalSource
            The constructed source class.
        """
        # Create nurbs object
        nurbs = GeomdlNURBS.Surface()

        # Get degree info
        nurbs.degree_u, nurbs.degree_v = (
            int(sg["Degree"].attrs["u"]),
            int(sg["Degree"].attrs["v"]),
        )

        # Set the control shape
        nurbs.ctrlpts_size_u = int(sg["Control Points"].attrs["u"])
        nurbs.ctrlpts_size_v = int(sg["Control Points"].attrs["v"])

        # Get the control points
        ctrlpts = np.zeros((nurbs.ctrlpts_size_u * nurbs.ctrlpts_size_v, 3))
        ctrlpts[:, 0] = sg["Control Points"]["u"][()]
        ctrlpts[:, 1] = sg["Control Points"]["v"][()]
        nurbs.ctrlpts = ctrlpts.tolist()

        # Get the weights
        nurbs.weights = sg["Weights"][()]

        # Get the knot vectors
        nurbs.knotsvector_u = sg["Knot Vectors"]["u"][()]
        nurbs.knotsvector_v = sg["Knot Vectors"]["v"][()]

        # Create source
        source_ctrlpts = sg["Control Points"]["s"][()]
        source = IsotropicInternalSource(
            source_ctrlpts.reshape(
                (source_ctrlpts.shape[0], nurbs.ctrlpts_size_u, nurbs.ctrlpts_size_v)
            )
        )
        source.set_patch(sg.attrs["Patch ID"], nurbs)
        source._converted = True

        return source
