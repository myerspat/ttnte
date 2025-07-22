import numpy as np
from geomdl import NURBS as GeomdlNURBS
from igakit.nurbs import NURBS as IgakitNURBS


class IsotropicInternalSource:
    """
    Isotropic internal source object. This defines an internal source using NURBS. This
    class evaluates.

    .. math::

        Q_{g} = \\sum_a\\sum_bR_{a,b}^{p,q}(\\hat{x}, \\hat{y})\\mathbf{Q}_{a,b}

    for a given :math:`(\\hat{x}, \\hat{y})` for each patch.
    """

    def __init__(self):
        """Initialize IsotropicInternalSource object."""
        self._converted = False
        self._patches = {}

    # ========================================================================
    # Public methods

    def add_patch(self, pid: int, patch: IgakitNURBS, source_ctrlpts: np.ndarray):
        """
        Add patch with an isotropic internal source.

        Parameters
        ----------
        pid: int
            Patch index corresponding to index in ``ttnte.iga.IGAMesh.patches``.
        patch: igakit.nurbs.NURBS
            Patch object for use in evaluation.
        source_ctrlpts: np.ndarray
            Control points for evaluating the source for the given patch with shape
            ``(ttnte.xs.Server.num_groups, ctrlpts_size_xhat, ctrlpts_size_yhat)``
            where ``ctrlpts_size_xhat`` are the control points along the ``xhat``
            parametric axis and ``yhat`` are the control points along the ``yhat`` axis.
            The following is evaluated for each patch
            .. math::

                Q_{g} = \\sum_a\\sum_bR_{a,b}^{p,q}(\\hat{x}, \\hat{y})\\mathbf{Q}_{a,b}
        """
        assert self._converted == False

        # Add patch to dictionary
        self._patches[pid] = (patch, source_ctrlpts)

    def igakit2geomdl(self):
        """Convert ``igakit.nurbs.NURBS`` to ``geomdl.NURBS.Surface``."""
        assert self._converted == False

        for pid, (patch, source_ctrlpts) in self._patches.items():
            # Initialize geomdl NURBS object
            geomdl_patch = GeomdlNURBS.Surface()

            # Set degree
            geomdl_patch.degree_u, geomdl_patch.degree_v = patch.degree

            # Set control points
            geomdl_patch.ctrlpts_size_u = patch.points.shape[0]
            geomdl_patch.ctrlpts_size_v = patch.points.shape[1]

            # Set control points
            geomdl_patch.ctrlpts = patch.points.reshape((-1, 3)).tolist()

            # Set weights
            geomdl_patch.weights = patch.weights.flatten().tolist()

            # Set knot vectors
            geomdl_patch.knotvector_u = patch.knots[0].tolist()
            geomdl_patch.knotvector_v = patch.knots[1].tolist()

            # Set patch
            self._patches[pid] = (geomdl_patch, source_ctrlpts)

    def evaluate_patch(self, pid: int, coords: np.ndarray):
        """
        Evaluate the internal source.

        Parameters
        ----------
        pid: int
            Patch index.
        coords: numpy.ndarray
            Parametric coordinates for evaluating the internal source.
            ``coords`` should be an array of shape ``(2, n)`` where
            ``n`` is the number of coordinates to evaluate.

        Returns
        -------
        source_data: numpy.ndarray or None
            Internal source of patch ``pid`` of shape
            ``(ttnte.xs.Server.num_groups, n)``. If ``pid`` is not
            defined as an isotropic internal source then returns ``None``.
        """
        assert self._converted and coords.ndim == 2 and coords.shape[0] == 2

        if pid in self._patches:
            # TODO: Write evaluation script for all groups using the patch
            raise NotImplementedError()

        else:
            return None
