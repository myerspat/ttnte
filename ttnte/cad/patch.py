import itertools
import multiprocessing as mp
from typing import List, Optional, Tuple, Union

import cotengra as ctg
import h5py as h5
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from geomdl.helpers import basis_functions, basis_functions_ders, find_spans
from igakit import cad
from igakit.nurbs import NURBS as IgakitNURBS

# Import geomdl
try:
    from geomdl.core import NURBS as GeomdlNURBS
except ImportError:
    print(
        "Geomdl was not compiled with cython. For best performance refer to"
        + "https://nurbs-python.readthedocs.io/en/5.x/install.html#compile"
        + "-with-cython"
    )
    from geomdl import NURBS as GeomdlNURBS

from ttnte.cad._utils import _igakit2geomdl
from ttnte.sources import IsotropicInternalSource


class Patch:
    """
    Patch object for handling a NURBS surface. This class includes parametric
    methods for basis evaluation, jacobian, etc and inverse mapping.

    Attributes
    ----------
    id: int
        Patch ID.
    name: str or None
        Name of the patch.
    material: str
        The patch's material fill.
    source: ttnte.sources.IsotropicInternalSource or None
        The patch's internal source
    shape: tuple of int
        Number of control points along each parametric dimension.
    ctrlpts: numpy.ndarray
        The control points of the patch of shape ``(*shape, 3)``.
    degree: tuple of int
        The degree of the basis functions along each parametric dimension.
    knotvectors: tuple of numpy.ndarray
        The knot vectors alone each parametric dimension.
    weights: numpy.ndarray
        The array of weight of shape ``shape``.
    bbox: list of tuple
        Bounding box - minimum and maximum points to contain the patch.
    """

    # Patch ID iterator
    _id_iter = itertools.count()
    centroids = [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]

    def __init__(
        self,
        nurbs: Union[IgakitNURBS, GeomdlNURBS.Surface],
        material: str,
        source: Optional[IsotropicInternalSource] = None,
        **kwargs,
    ):
        """
        Initialize Patch object.

        Parameters
        ----------
        nurbs: igakit.nurbs.NURBS
            NURBS surface.
        material: str
            Material fill of the patch.
        source: ttnte.sources.IsotropicInternalSource or None
            Internal source of the patch.
        name: str or None
            The name of the patch.
        max_processes: int, default=multiprocessing.cpu_count() - 1
            The maximum number of processes allowed at one time.
        """
        self._nurbs = nurbs
        self._material = material
        self._source = source

        # Unpack kwargs
        self._name = kwargs.get("name", None)
        self._max_processes = kwargs.get("max_processes", mp.cpu_count() - 1)

        # Create ID
        self._id = next(self._id_iter)

        # Boundary orientations
        self._orientations = [1, 1, 1, 1]

        # Make the source independent
        if self._source is not None:
            self._source.set_patch(pid=self._id, nurbs=self._nurbs.copy())

    def set_source(self, source: IsotropicInternalSource):
        """
        Set the internal source.

        Parameters
        ----------
        source: ttnte.sources.IsotropicInternalSource or None
            Internal source of the patch.
        """
        assert isinstance(self._nurbs, IgakitNURBS)
        self._source = source
        self._source.set_patch(pid=self._id, nurbs=self._nurbs.copy())

    # ========================================================================
    # Mesh methods
    def refine(self, factor: Union[int, List[int]], degree: Union[int, List[int]]):
        """
        Refine the patch's mesh.

        Parameters
        ----------
        factor: int or list of two ints
            Number of elements or knot spans along each parametric axis.
            If an integer is given then that is applied to both parametric
            axes.
        degree: ind or list of two ints
            Degree of NURBS surface along each parametric axis. If an
            integer is given then that is applied to both parametric axes.
        """
        assert isinstance(self._nurbs, IgakitNURBS)

        # Run igakit method to refine
        self._nurbs = cad.refine(self._nurbs, factor=factor, degree=degree)

    def igakit2geomdl(self):
        """Convert ``igakit.nurbs.NURBS`` to ``geomdl.NURBS.Surface``."""
        assert isinstance(self._nurbs, IgakitNURBS)

        # Convert patch's NURBS
        self._nurbs = _igakit2geomdl(self._nurbs)

        # Convert source
        if self._source:
            self._source.igakit2geomdl()

    def initialize_boundaries(self, decimals: int = 8):
        """
        Determine the orientation of the normals at each boundary and compute
        the physical coordinates of the centroid at each boundary.

        Parameters
        ----------
        decimals: int, default=8
            Number of decimal points to round each physical coordinate to.

        Returns
        -------
        Points: dict
            Dictionary of centroid evaluations where the key is either the ID of
            this patch or ``None`` depending on if the boundary exists in
            physical space. The value is the physical coordinate of the centroid
            itself.
        """
        # Get the orientations of the boundaries
        self._get_orientations()

        # Centroids in physical space
        points = {}
        for centroid, orientation in zip(self.centroids, self._orientations):
            points[tuple(np.round(np.array(self(centroid)[:-1]), decimals))] = (
                self._id if orientation != 0 else None
            )

        return points

    def _get_orientations(self):
        for centroid in self.centroids:
            self._boundary_orientation(centroid)

    def _boundary_orientation(self, centroid: Tuple[float]):
        # Get boundary index
        bidx = self.get_boundary_idx(centroid)

        # Test points
        coords = np.zeros((3, 2)) if 0 in centroid else np.ones((3, 2))
        coords[:, 0 if centroid[0] == 0.5 else 1] = np.array([0.25, 0.5, 0.75])

        # Evaluate points
        points, normals = self.normal(centroid, np.array([0.25, 0.5, 0.75]))

        # Check if boundary is a point
        if np.isclose(points, points[0]).all():
            self._orientations[bidx] = 0

        else:
            # Change to comparison points
            coords = 0.5 * np.ones((3, 2))
            coords[:, 0 if centroid[0] == 0.5 else 1] = np.array([0.25, 0.5, 0.75])

            # Evaluate points
            comp_points = self(coords)[:, :-1]

            # Calculate vectors
            vecs = comp_points - points

            self._orientations[bidx] = (
                1 if ((vecs * normals).sum(axis=1) < 0).any() else -1
            )

    @staticmethod
    def get_boundary_idx(centroid: Tuple[float]):
        """
        Get the boundary index for a given centroid from
        ``ttnte.cad.Patch.centroids``.

        Parameters
        ----------
        centroid: One of (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)
            The center in parametric space of a boundary.
        """
        for bidx in range(len(Patch.centroids)):
            if centroid == Patch.centroids[bidx]:
                return bidx

        raise RuntimeError(f"Centroid provided does not match one of {Patch.centroids}")

    # ========================================================================
    # Parametric methods
    def find_spans(self, param_idx: int, coords: np.ndarray):
        """
        Compute the knot spans along a parametric dimension for a vector of parametric
        coordinates.

        Parameters
        ----------
        param_idx: int
            Which parametric axis.
        coords: numpy.ndarray
            Coordinates to find the knot spans. ``coords`` should be an array
            of shape ``(n,)`` where ``n`` is the number of coordinates
            to evaluate.

        Returns
        -------
        spans: numpy.ndarray
            Index in the knotvector for the start of the knot span that
            the given coordinate is located in.
        """
        coords = np.array(coords, dtype=float)
        assert coords.ndim == 1 and isinstance(self._nurbs, GeomdlNURBS.Surface)

        # Find knot spans from knotvector
        return np.array(
            find_spans(
                int(self.degree[param_idx]),
                self.knotvectors[param_idx],
                int(self.shape[param_idx]),
                coords,
            )
        )

    def basis_functions(self, coords: np.ndarray):
        """
        Compute all non-vanishing NURBS basis functions. The current implementation
        assumes an open knot vector with single multiplicity on inner knots.

        Parameters
        ----------
        coords: numpy.ndarray
            Coordinates to evaluate the basis functions. ``coords`` should be an
            array of shape ``(n, 2)`` where ``n`` is the number of coordinates
            to evaluate.

        Returns
        -------
        basis_data: numpy.ndarray
            Non-zero basis function evaluations of shape ``(n, p + 1, q + 1)``
            where ``p`` and ``q`` are the degrees of the basis functions along
            each respective parametric dimension.
        """
        assert (
            coords.ndim == 2
            and coords.shape[1] == 2
            and isinstance(self._nurbs, GeomdlNURBS.Surface)
        )
        coords = coords.numpy() if isinstance(coords, tn.Tensor) else coords

        # Get weights
        weights = self.weights

        # Number of dimensions
        ndim = len(self.degree)

        # Compute B-Spline basis
        spans = [[] for _ in range(ndim)]
        basis = [[] for _ in range(ndim)]

        for i in range(ndim):
            # Handle cases when we're evaluating at exactly 1
            spans[i] = (len(self.knotvectors[i]) - 2 * self.degree[i]) * np.ones(
                coords.shape[0], dtype=int
            )
            basis[i] = np.zeros((coords.shape[0], self.degree[i] + 1))
            basis[i][:, -1] = 1

            # Calculate cases when the coordinate < 1
            mask = coords[:, i] != 1
            if mask.any():
                spans[i][mask] = np.array(self.find_spans(i, coords[mask, i]))
                basis[i][mask, :] = np.array(
                    basis_functions(
                        self.degree[i],
                        self.knotvectors[i],
                        spans[i][mask],
                        coords[mask, i],
                    )
                )

        # Evaluate numerator of 2-D NURBS basis
        basis_data = ctg.einsum("ij,ik->ijk", basis[0], basis[1]) * np.array(
            list(
                map(
                    lambda i: weights[
                        spans[0][i] - self.degree[0] : spans[0][i] + 1,
                        spans[1][i] - self.degree[1] : spans[1][i] + 1,
                    ],
                    np.arange(coords.shape[0]),
                ),
            )
        )

        # Evaluate denominator and return
        return basis_data / np.sum(np.sum(basis_data, axis=-1), axis=-1).reshape(
            (-1, 1, 1)
        )

    def basis_function_grads(self, coords: np.ndarray, order: int = 1):
        """
        Compute gradients of the basis functions for a given patch. The current
        implementation assumes an open knot vector with single multiplicity on inner
        knots.

        Parameters
        ----------
        coords: numpy.ndarray
            Coordinates to evaluate the gradients of the basis functions. ``coords``
            should be an array of shape ``(n, 2)`` where ``n`` is the number of
            coordinates to evaluate.
        order: int, default=1
            Derivative order to be evaluated.

        Returns
        -------
        basis_data: numpy.ndarray
            Basis data of shape ``(n, p + 1, q + 1, 3)`` where ``p`` and ``q`` are the
            degrees of the basis functions for each parametric dimension.
            ``(:, :, :, 0)`` contains non-vanishing basis function evaluations,
            ``(:, :, :, 1)`` contains non-vanishing basis function derivatives with
            respect to ``u`` and ``(:, :, :, 2)`` contains non-vanishing basis function
            derivatives with respect to ``v``.
        """
        assert (
            coords.ndim == 2
            and coords.shape[1] == 2
            and isinstance(self._nurbs, GeomdlNURBS.Surface)
        )

        # Get weights
        weights = self.weights

        # Number of dimensions
        ndim = len(self.degree)

        # Compute B-Spline basis
        spans = [[] for _ in range(ndim)]
        basis = [[] for _ in range(ndim)]
        basis_ders = [[] for _ in range(ndim)]

        for i in range(ndim):
            spans[i] = np.array(self.find_spans(i, coords[:, i]))
            evals = np.array(
                basis_functions_ders(
                    self.degree[i],
                    self.knotvectors[i],
                    spans[i],
                    coords[:, i],
                    order,
                )
            )

            basis[i] = evals[:, 0, :]
            basis_ders[i] = evals[:, 1, :]

        basis_data = np.zeros(
            (coords.shape[0], self.degree[0] + 1, self.degree[1] + 1, 3)
        )

        # Compute numerator of basis functions
        weights = np.array(
            list(
                map(
                    lambda i: weights[
                        spans[0][i] - self.degree[0] : spans[0][i] + 1,
                        spans[1][i] - self.degree[1] : spans[1][i] + 1,
                    ],
                    np.arange(coords.shape[0]),
                ),
            )
        )
        basis_data[..., 0] = ctg.einsum("ij,ik->ijk", basis[0], basis[1]) * weights

        # Compute denominator
        denom = np.sum(np.sum(basis_data[..., 0], axis=-1), axis=-1).reshape((-1, 1, 1))

        # Scale basis functions by denominator
        basis_data[..., 0] /= denom

        # Compute first term of quotient rule
        basis_data[..., 1] += (
            ctg.einsum("ij,ik->ijk", basis_ders[0], basis[1]) * weights / denom
        )
        basis_data[..., 2] += (
            ctg.einsum("ij,ik->ijk", basis[0], basis_ders[1]) * weights / denom
        )

        # Compute second term of quotient rule
        basis_data[..., 1] -= basis_data[..., 0] * np.sum(
            np.sum(basis_data[..., 1], axis=-1), axis=-1
        ).reshape((-1, 1, 1))
        basis_data[..., 2] -= basis_data[..., 0] * np.sum(
            np.sum(basis_data[..., 2], axis=-1), axis=-1
        ).reshape((-1, 1, 1))

        return basis_data

    def jacobian(self, coords: np.ndarray):
        """
        Computes the Jacobian matrix :math:`\\frac{\\partial(x, y)}{\\partial(\\hat{x},
        \\hat{y})}` for the pull back from the parametric domain to the physical.

        Parameters
        ----------
        coords: numpy.ndarray
            Parametric coordinates to evaluate the Jacobian at. ``coords``
            should be an array of shape ``(n, 2)`` where ``n`` is the number of
            coordinates to evaluate.

        Returns
        -------
        jacobian: numpy.ndarray
            Jacobian matrix for each set of parametric coordinates or shape
            ``(n, 2, 2)``.
        """
        assert (
            coords.ndim == 2
            and coords.shape[1] == 2
            and isinstance(self._nurbs, GeomdlNURBS.Surface)
        )

        # Get spans
        spans = np.concatenate(
            [
                self.find_spans(0, coords[:, 0])[..., np.newaxis] - self.degree[0],
                self.find_spans(1, coords[:, 1])[..., np.newaxis] - self.degree[1],
            ],
            axis=-1,
        )

        # Calculate basis functions
        dR = self.basis_function_grads(coords)[..., 1:]

        # Get control points
        ctrlpts = self.ctrlpts[..., :-1]

        # Create empty array to fill
        J = np.empty((spans.shape[0], 2, 2), dtype=np.float64)

        for i in range(spans.shape[0]):
            J[i, ...] = ctg.einsum(
                "abc,abd->cd",
                dR[i, ...],
                ctrlpts[
                    spans[i, 0] : spans[i, 0] + self.degree[0] + 1,
                    spans[i, 1] : spans[i, 1] + self.degree[1] + 1,
                    :,
                ],
            )

        return J

    def normal(self, centroid: tuple, coords: np.ndarray):
        """
        Calculate the outward normal vector at parametric coordinates on the
        boundary.

        Parameters
        ----------
        centroid: (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5),
            Parametric coordinate of center of boundary.
        coords: numpy.ndarray
            Parametric coordinates to evaluate the normals. ``coords``
            should be an array of shape ``(n,)`` where ``n`` is the number of
            coordinates to evaluate.

        Returns
        -------
        points: numpy.ndarray
            Locations in physical space of normal vectors of shape ``(n, 2)``.
        normals: numpy.ndarray
            Normal vectors of unit length of shape ``(n, 2)``.
        """
        assert isinstance(self._nurbs, GeomdlNURBS.Surface) and coords.ndim == 1

        # Fill arrays
        points = np.zeros((coords.shape[0], 2))
        normals = np.zeros((coords.shape[0], 2))

        constant = 0 if centroid[0] + centroid[1] == 0.5 else 1
        bidx = self.get_boundary_idx(centroid)

        # Function and indices to compute the derivative
        a, b, sign, calc_derivative = (
            (
                1,
                0,
                -1 if constant == 1 else 1,
                lambda i: self._nurbs.derivatives(coords[i], constant, order=1),
            )
            if bidx < 2
            else (
                0,
                1,
                -1 if constant == 0 else 1,
                lambda i: self._nurbs.derivatives(constant, coords[i], order=1),
            )
        )

        # Iterate over each coordinate
        for i in range(coords.shape[0]):
            # Calculate derivatives
            dd = np.array(calc_derivative(i))

            # Save point in (x, y)
            points[i, :] = dd[0, 0, :-1]

            # Compute normal
            normal = dd[a, b, :-1][::-1]
            if (normal != 0).any():
                normal[0] *= -1
                normal /= sign * np.linalg.norm(normal, 2)

            # Save normal vector
            normals[i, :] = self._orientations[bidx] * normal

        return points, normals

    def evaluate(self, coords: Union[np.ndarray, Tuple[float], List[float]]):
        """
        Evaluate the NURBS function given the parametric coordinates.

        Parameters
        ----------
        coords: numpy.ndarray or array like of length 2
            Parametric coordinates to evaluate the NURBS. ``coords`` should be an
            array of shape ``(n, 2)`` where ``n`` is the number of samples or
            an individual coordinate of length 2.

        Returns
        -------
        Points: numpy.ndarray or array like of length 2
            Results of evaluation of shape ``(n, 3)`` or ``(3,)``.
        """
        return self(coords)

    def inverse_map(
        self,
        physical_coords: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8,
        coords0: Optional[np.ndarray] = None,
    ):
        """
        Map coordinates in the physical domain to the parametric domain.

        Parameters
        ----------
        physical_coords: numpy.ndarray
            Array or coordinates of shape ``(n, 2)`` where ``n`` is the
            number of coordinates. ``physical_coords[:, 0]`` are the ``x``
            positions and ``physical_coords[:, 1]`` are the ``y`` positions.
        max_iter: int, default=100
            Maximum number of Newton-Raphson iterations for each viable
            patch.
        tol: float, default=1e-8
            Tolerance of inverse map computed with Newton-Raphson.
        coords0: np.ndarray, default=None
            Initial guess.

        Returns
        -------
        coords: numpy.ndarray
            Physical coordinates of shape ``(n, 2)``.
        converged: bool
            Coordinates converged within the tolerance.
        """
        assert physical_coords.ndim == 2 and physical_coords.shape[1] == 2

        # Fill array
        coords = 0.5 * np.ones(physical_coords.shape) if coords0 is None else coords0

        # Distances
        old_distances = np.ones(physical_coords.shape[0])
        new_distances = old_distances.copy()

        # Set all to unconverged
        unconverged = np.ones(physical_coords.shape[0]).astype(bool)

        for _ in range(max_iter):
            # Calculate Jacobian
            jacobian = self.jacobian(coords[unconverged, :])
            jacobian[:, 1, 0] *= -1
            jacobian[:, 0, 1] *= -1
            jacobian[:, 0, 0], jacobian[:, 1, 1] = (
                jacobian[:, 1, 1].copy(),
                jacobian[:, 0, 0].copy(),
            )

            # Calculate determinant
            determinant = np.zeros(physical_coords.shape[0])
            determinant[unconverged] = (
                jacobian[:, 0, 0] * jacobian[:, 1, 1]
                - jacobian[:, 0, 1] * jacobian[:, 1, 0]
            )

            # Mask out determinants of zero
            mask = unconverged & (determinant != 0)

            # Calculate new coordinates
            if mask.any():
                coords[mask, :] -= (
                    1
                    / determinant[(determinant != 0)].reshape((-1, 1))
                    * ctg.einsum(
                        "abc,ab->ac",
                        jacobian[(determinant[unconverged] != 0), ...],
                        self(coords[mask, :])[:, :-1] - physical_coords[mask, :],
                    )
                )

                # Check if any updates are outside the contraints
                coords[coords < 0] = 0
                coords[coords > 1] = 1

                # Calculate new distances
                new_distances[mask] = np.linalg.norm(
                    self(coords[mask, :])[:, :-1] - physical_coords[mask, :], 2, axis=-1
                )

                # Update what has converged
                dist_change = np.zeros(physical_coords.shape[0])
                dist_change[old_distances != 0] = (
                    np.abs(
                        new_distances[old_distances != 0]
                        - old_distances[old_distances != 0]
                    )
                    / old_distances[old_distances != 0]
                )
                unconverged = (old_distances >= tol) & (dist_change > (tol * 1e-3))

                # Check if the convergence is finished
                if (unconverged == False).all():
                    return coords, True

                # Update old distances
                old_distances[mask] = new_distances[mask]

            else:
                break

        return coords, False

    # ========================================================================
    # Plotting

    def plot_normals(self, num_nodes=256):
        """
        Plot normals at all boundaries of a given patch.

        Parameters
        ----------
        num_nodes: int, default=256
            Number of positions to sample the mesh for each patch.

        Returns
        -------
        ax: matplotlib.axes.Axes
            Resulting matplotlib axis object.
        """
        # Create axis
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Get points
        X, Y = np.meshgrid(np.linspace(0, 1, num_nodes), np.linspace(0, 1, num_nodes))
        points = self(
            np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)
        ).reshape((num_nodes, num_nodes, 3))[..., :-1]
        del X, Y

        # Plot surface
        ax.plot_surface(
            points[..., 0], points[..., 1], 1 * np.zeros((num_nodes, num_nodes))
        )

        # Iterate through boundaries
        labels = ["$\\hat y = 0$", "$\\hat y = 1$", "$\\hat x = 0$", "$\\hat x = 1$"]
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, centroid in enumerate(self.centroids):
            # Calculate normals
            points, normals = self.normal(centroid, np.linspace(0, 1, 12)[1:-1])

            # Plot normals
            ax.quiver(
                points[:, 0],
                points[:, 1],
                np.zeros(10),
                normals[:, 0],
                normals[:, 1],
                np.zeros(10),
                label=labels[i],
                color=colors[i],
                length=0.1,
            )
        # Label axes
        ax.set_xlabel("$x~(cm)$")
        ax.set_ylabel("$y~(cm)$")
        ax.set_aspect("equal")
        ax.view_init(elev=90, azim=-90, roll=0)
        return ax
        return (
            np.array(self._nurbs.weights)
            if isinstance(self._nurbs, GeomdlNURBS.Surface)
            else self._nurbs.weights
        )

    # ========================================================================
    # Overloads
    def __str__(self):
        backend = "geomdl" if isinstance(self._nurbs, GeomdlNURBS.Surface) else "igakit"
        return (
            f"Patch(material={self._material}, source={self._source}, "
            + f"id={self._id}, name={self._name}, shape={self.shape}, backend={backend})"
        )

    def __repr__(self):
        return self.__str__()

    def __call__(self, coords: Union[np.ndarray, Tuple[float], List[float]]):
        """
        Evaluate the NURBS function given the parametric coordinates.

        Parameters
        ----------
        coords: numpy.ndarray or array like of length 2
            Parametric coordinates to evaluate the NURBS. ``coords`` should be an
            array of shape ``(n, 2)`` where ``n`` is the number of samples or
            an individual coordinate of length 2.

        Returns
        -------
        Points: numpy.ndarray or array like of length 2
            Results of evaluation of shape ``(n, 3)`` or ``(3,)``.
        """
        assert isinstance(self._nurbs, GeomdlNURBS.Surface)

        if isinstance(coords, np.ndarray):
            assert coords.ndim == 2 and coords.shape[1] == 2

            # Evaluate array
            result = np.array(self._nurbs.evaluate_list(coords))
            return result

        else:
            assert len(coords) == 2

            # Evaluate single coordinate
            return type(coords)(self._nurbs.evaluate_single(coords))

    # ========================================================================
    # I / O

    def save(self, mg: h5.Group):
        """
        Save patch information

        Parameters
        ----------
        mg: h5py.Group
            Mesh group to append patch information to.

        Returns
        -------
        mg: h5py.Group
            Mesh group to append patch information to.
        """
        assert isinstance(self._nurbs, GeomdlNURBS.Surface)

        # Create group for patch
        pg = mg.create_group(f"Patch {self.id}")
        pg.attrs["Material"] = self.material
        pg.attrs["ID"] = self.id
        pg.attrs["Max Processes"] = self._max_processes
        if self.name is not None:
            pg.attrs["Name"] = self.name

        # Save degree
        dg = pg.create_group("Degree")
        dg.attrs["u"], dg.attrs["v"] = self.degree

        # Save control points
        cp = pg.create_group("Control Points")
        cp.attrs["u"], cp.attrs["v"] = self.shape
        cp.create_dataset("u", data=self.ctrlpts.reshape((-1, 3))[:, 0])
        cp.create_dataset("v", data=self.ctrlpts.reshape((-1, 3))[:, 1])

        # Save weights
        pg.create_dataset("Weights", data=self.weights.flatten())

        # Save knot vectors
        kg = pg.create_group("Knot Vectors")
        kg.create_dataset("u", data=self.knotvectors[0])
        kg.create_dataset("v", data=self.knotvectors[1])

        # Boundary orientation
        pg.create_dataset("Boundary Orientations", data=self._orientations)

        # Save source
        if self._source is not None:
            pg = self._source.save(pg)

        return mg

    @classmethod
    def load(cls, pg: h5.Group):
        """
        Load patch information.

        Parameters
        ----------
        pg: h5py.Group
            Patch group to unpack info from.

        Returns
        -------
        patch: ttnte.cad.Patch
            The patch.
        """
        # Create nurbs object
        nurbs = GeomdlNURBS.Surface()

        # Get degree info
        nurbs.degree_u, nurbs.degree_v = (
            int(pg["Degree"].attrs["u"]),
            int(pg["Degree"].attrs["v"]),
        )

        # Set the control shape
        nurbs.ctrlpts_size_u = int(pg["Control Points"].attrs["u"])
        nurbs.ctrlpts_size_v = int(pg["Control Points"].attrs["v"])

        # Get the control points
        ctrlpts = np.zeros((nurbs.ctrlpts_size_u * nurbs.ctrlpts_size_v, 3))
        ctrlpts[:, 0] = pg["Control Points"]["u"][()]
        ctrlpts[:, 1] = pg["Control Points"]["v"][()]
        nurbs.ctrlpts = ctrlpts.tolist()

        # Get the weights
        nurbs.weights = pg["Weights"][()]

        # Get the knot vectors
        nurbs.knotsvector_u = pg["Knot Vectors"]["u"][()]
        nurbs.knotsvector_v = pg["Knot Vectors"]["v"][()]

        # Create patch
        patch = cls(nurbs, pg.attrs["Material"])
        patch._id = pg.attrs["ID"]
        patch._name = pg.attrs["Name"] if "Name" in pg.attrs else None
        patch._max_processes = pg.attrs["Max Processes"]
        patch._orientations = pg["Boundary Orientations"][()]
        patch._source = (
            IsotropicInternalSource.load(pg["Source"]) if "Source" in pg else None
        )

        return patch

    # ========================================================================
    # Getters / Setters

    def set_phi(self, phi: np.ndarray):
        """
        Set the scalar flux control variables.

        Parameters
        ----------
        phi: numpy.ndarray
            Control variables of NURBS expansion for scalar flux. The array
            should have more than one dimensions with the first dimension
            indicating which patch the control variables belong to.
        """
        assert isinstance(self._nurbs, GeomdlNURBS.Surface)

        new_ctrlpts = self._nurbs.ctrlpts
        for i in range(phi.shape[-1]):
            new_ctrlpts[i] = [*new_ctrlpts[i][:-1], phi[i]]
        self._nurbs.ctrlpts = new_ctrlpts

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def material(self):
        return self._material

    @property
    def source(self):
        return self._source

    @property
    def shape(self):
        return (
            (self._nurbs.ctrlpts_size_u, self._nurbs.ctrlpts_size_v)
            if isinstance(self._nurbs, GeomdlNURBS.Surface)
            else self._nurbs.shape
        )

    @property
    def ctrlpts(self):
        return (
            np.array(self._nurbs.ctrlpts).reshape((*self.shape, 3))
            if isinstance(self._nurbs, GeomdlNURBS.Surface)
            else self._nurbs.points
        )

    @property
    def degree(self):
        return self._nurbs.degree

    @property
    def knotvectors(self):
        return (
            (np.array(self._nurbs.knotvector_u), np.array(self._nurbs.knotvector_v))
            if isinstance(self._nurbs, GeomdlNURBS.Surface)
            else self._nurbs.knots
        )

    @property
    def weights(self):
        return np.array(self._nurbs.weights).reshape(self.shape)

    @property
    def bbox(self):
        assert isinstance(self._nurbs, GeomdlNURBS.Surface)
        return self._nurbs.bbox
