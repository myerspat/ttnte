from typing import List, Literal, Union

import cotengra as ctg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy
from geomdl import NURBS
from geomdl.helpers import basis_functions, basis_functions_ders, find_spans
from igakit import cad

from .boundary import Boundary


class IGAMesh(object):
    """
    IGA Mesh object. This object handles all NURBS definitions, basis function
    evaluations, and connectivity.

    Attributes
    ----------
    num_patches: int
        Number of patches in the mesh
    degree: list
        The degree for each parametric axis of all patches.
        This assumes all patches have the same degree.
    ndim: int
        Number of spatial dimensions.
    """

    def __init__(self, patches: dict):
        """
        Initialize IGAMesh object.

        .. warning::
           There must be no hanging nodes between patches each
           defined with open knot vectors and internal knot multiplicity
           of one.

        Parameters
        ----------
        patches: dict of igakit.nurbs.NURBS and str
            Patches in the IGA mesh. The keys are the patches
            defined as ``igakit.nurbs.NURBS`` and the values
            are the names of their materials.
        """
        # Get information
        self.patches = list(patches.keys())
        self._materials = list(patches.values())

        # Interfaces and boundaries
        self._incident_boundaries = {p: 4 * [None] for p in range(self.num_patches)}

        # Boundaries
        self._bcs_set = False

        # States
        self._finalized = False
        self._connected = False

    def connect(self, decimals=8):
        """
        Connect patches. Determine patch interfaces for passing boundary conditions.
        Patches not connected are assumed to be vacuum boundary conditions unless
        otherwise specified in :meth:`ttnte.iga.IGAMesh.set_reflective_condition`.

        Parameters
        ----------
        decimals: int, default=8
            Number of decimals to round boundary evaluations. These
            are matched with other patch boundaries to determine
            which share boundaries.
        """
        assert not self._finalized and not self._connected
        self._decimals = decimals

        # Iterate through patches
        for i in range(self.num_patches):
            patch = self.patches[i]

            # Initialize geomdl NURBS object
            geomdl_patch = NURBS.Surface()

            # Set degree
            geomdl_patch.degree_u, geomdl_patch.degree_v = patch.degree

            # Set material
            geomdl_patch.name = self._materials[i]

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
            self.patches[i] = geomdl_patch

        del self._materials

        # Find all boundaries and build spatial hash table
        self._boundary_hash = {}
        for p, patch in enumerate(self.patches):
            # Evaluate boundary centers
            for boundary_idx, coord in enumerate(
                [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]
            ):
                # Check the normal is non-zero
                point, normal = self.normal(p, np.array(coord).reshape((2, 1)))
                if np.sqrt(np.sum(normal**2)) != 0:
                    # Get physical position
                    point = tuple(np.round(point.flatten(), decimals))

                    # Create boundary
                    boundary = Boundary(p, 1, boundary_idx)

                    # Add boundary
                    if point in self._boundary_hash:
                        self._boundary_hash[point].append(boundary)
                    else:
                        self._boundary_hash[point] = [boundary]

        # Check each face is connected to at most 1 other patch
        for boundaries in self._boundary_hash.values():
            if len(boundaries) > 2:
                raise RuntimeError(
                    "The boundary of each patch should have at most one other neighbor"
                )

        self._connected = True

    def finalize(self):
        """Finalize mesh."""
        self._finalized = True

    # ========================================================================
    # Boundary condition methods

    def set_reflective_condition(
        self,
        faces: Union[
            tuple,
            Literal["left", "right", "bottom", "top"],
        ],
    ):
        """
        Set reflective boundary conditions. This method determines which patches have
        reflective boundaries based on the face.

        .. warning::
           All reflective boundaries must be axis-aligned.

        Parameters
        ----------
        faces: tuple of "left", "right", "bottom", "top"
            Which axis-aligned faces to set as boundary conditions.
        """
        if not self._connected or self._finalized:
            raise RuntimeError("Patches must be connected but not finalized")

        faces = set(np.array([faces]).flatten())

        # Get all boundary locations
        centers = np.array(list(self._boundary_hash.keys()))

        for face in faces:
            if face not in ["left", "right", "top", "bottom"]:
                raise RuntimeError(
                    "Boundary must be one of 'left', 'right', 'bottom', or 'top'"
                )
            # Find locations based on side (assumes axis aligned)
            boundaries = {
                "left": lambda c: c[c[:, 0] == c[:, 0].min(), :],
                "right": lambda c: c[c[:, 0] == c[:, 0].max(), :],
                "bottom": lambda c: c[c[:, 1] == c[:, 1].min(), :],
                "top": lambda c: c[c[:, 1] == c[:, 1].max(), :],
            }[face](centers)

            # Append the same boundary
            for i in range(boundaries.shape[0]):
                self._boundary_hash[tuple(boundaries[i, :])] += self._boundary_hash[
                    tuple(boundaries[i, :])
                ]

    def get_connected_patch(self, p: int, coord: tuple):
        """
        Get the patch a specific patch's boundary connects to.

        .. warning::
           There must be no hanging nodes between patches;
           therefore, exactly one boundary of the given patch
           connects to another.

        Parameters
        p: int
            Index of patch.
        coord: (0, 0.5), (1, 0.5), (0.5, 0), or (0.5, 1)
            Parametric coordinate of center of boundary.

        Returns
        connected_p: int or None
            Index of the connected patch. If it is ``None`` then this
            boundary is a vacuum boundary condition.
        """
        if not self._connected or not self._finalized:
            raise RuntimeError("Patches must be connected and finalized")

        # Calculate physical point
        point, normal = self.normal(p, np.array(coord).reshape((-1, 1)))
        point = tuple(np.round(point.flatten(), self._decimals))

        # Check boundary exists
        if (normal == 0).all():
            return None

        # Check point in spatial hash
        if point in self._boundary_hash:
            ps = np.array([b.p for b in self._boundary_hash[point]])

            # Handle vacuume boundary condition
            if ps.size == 1:
                return None
            elif (ps != p).any():
                # Return adjacent patch
                return ps[ps != p][0]
            else:
                # Handle reflective boundary condition
                return p

    def get_orientation(self, p: int, const_param_idx: int, const_param: int):
        """"""
        # Get current patch
        patch = self.patches[p]

        # Get end points in parametric space
        points = np.zeros((2, 2)) if const_param == 0 else np.ones((2, 2))
        points[abs(const_param_idx - 1), abs(const_param - 1)] = abs(const_param - 1)

        # Evaluate the tangent vectors
        left = np.array(patch.derivatives(points[0, 0], points[1, 0], order=1))
        right = np.array(patch.derivatives(points[0, 1], points[1, 1], order=1))

        dleft, dright = (
            (left[1, 0, :-1], right[1, 0, :-1])
            if abs(const_param_idx - 1) > 0
            else (left[0, 1, :-1], right[0, 1, :-1])
        )

        return 1 if np.cross(dleft, dright) > 0 else -1

    # ========================================================================
    # Refinement methods

    def refine(
        self, p: int, factor: Union[int, List[int]], degree: Union[int, List[int]]
    ):
        """
        Refine the mesh of a given patch.

        Parameters
        ----------
        p: int
            Patch index.
        factor: int or list of two ints
            Number of elements or knot spans along each parametric axis.
            If an integer is given then that is applied to both parametric
            axes.
        degree: ind or list of two ints
            Degree of NURBS surface along each parametric axis. If an
            integer is given then that is applied to both parametric axes.
        """
        assert not self._finalized

        # Refine patch
        self.patches[p] = cad.refine(self.patches[p], factor=factor, degree=degree)

    # ========================================================================
    # Parametric methods

    def find_spans(self, p: int, param_idx: int, coords: np.ndarray):
        """
        Compute the knot spans along a parametric dimension for a vector of parametric
        coordinates.

        Parameters
        ----------
        p: int
            Index of patch.
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
        assert coords.ndim == 1

        # Find knot spans from knotvector
        return np.array(
            find_spans(
                self.degree[param_idx],
                self.patches[p].knotvector[param_idx],
                self.patches[p].ctrlpts_size,
                coords,
            )
        )

    def basis_functions(self, p: int, coords: np.ndarray):
        """
        Compute all non-vanishing 2-D NURBS basis functions. The current implementation
        assumes an open knot vector with single multiplicity on inner knots.

        Parameters
        ----------
        p: int
            Index of patch.
        coords: numpy.ndarray
            Coordinates to evaluate the basis functions. ``coords`` should be an
            array of shape ``(2, n)`` where ``n`` is the number of coordinates
            to evaluate.

        Returns
        -------
        basis_data: numpy.ndarray
            Non-zero basis function evaluations of shape ``(n, p + 1, q + 1)``
            where ``p`` and ``q`` are the degrees of the basis functions along
            each respective parametric dimension.
        """
        assert coords.ndim == 2 and coords.shape[0] == 2

        # Get weights
        weights = np.array(self.patches[p].weights).reshape(
            (
                self.patches[p].ctrlpts_size_u,
                self.patches[p].ctrlpts_size_v,
            )
        )

        # Compute B-Spline basis
        spans = [[] for _ in range(self.ndim)]
        basis = [[] for _ in range(self.ndim)]

        for i in range(self.ndim):
            # Handle cases when we're evaluating at exactly 1
            spans[i] = (
                len(self.patches[p].knotvector[i]) - 2 * self.degree[i]
            ) * np.ones(coords.shape[1], dtype=int)
            basis[i] = np.zeros((coords.shape[1], self.degree[i] + 1))
            basis[i][:, -1] = 1

            # Calculate cases when the coordinate < 1
            if (coords[i, :] != 1).any():
                spans[i][coords[i, :] != 1] = np.array(
                    self.find_spans(p, i, coords[i, :][coords[i, :] != 1])
                )
                basis[i][coords[i, :] != 1, :] = np.array(
                    basis_functions(
                        self.degree[i],
                        self.patches[p].knotvector[i],
                        spans[i][coords[i, :] != 1],
                        coords[i, :][coords[i, :] != 1],
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
                    np.arange(coords.shape[1]),
                ),
            )
        )

        # Evaluate denominator and return
        return basis_data / np.sum(np.sum(basis_data, axis=-1), axis=-1).reshape(
            (-1, 1, 1)
        )

    def basis_function_grads(self, p: int, coords: np.ndarray, order: int = 1):
        """
        Compute gradients of the basis functions for a given patch. This implementation
        assumes open knot vectors with all interior knots of single multiplicity.

        Parameters
        ----------
        p: int
            Index of patch.
        coords: numpy.ndarray
            Coordinates to evaluate the gradients of the basis functions. ``coords``
            should be an array of shape ``(2, n)`` where ``n`` is the number of
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
        assert coords.ndim == 2 and coords.shape[0] == 2

        # Get weights
        weights = np.array(self.patches[p].weights).reshape(
            (
                self.patches[p].ctrlpts_size_u,
                self.patches[p].ctrlpts_size_v,
            )
        )

        # Compute B-Spline basis
        spans = [[] for _ in range(self.ndim)]
        basis = [[] for _ in range(self.ndim)]
        basis_ders = [[] for _ in range(self.ndim)]

        for i in range(self.ndim):
            spans[i] = np.array(self.find_spans(p, i, coords[i, :]))
            evals = np.array(
                basis_functions_ders(
                    self.degree[i],
                    self.patches[p].knotvector[i],
                    spans[i],
                    coords[i, :],
                    order,
                )
            )

            basis[i] = evals[:, 0, :]
            basis_ders[i] = evals[:, 1, :]

        basis_data = np.zeros(
            (coords.shape[1], self.degree[0] + 1, self.degree[1] + 1, 3)
        )

        # Compute numerator of basis functions
        weights = np.array(
            list(
                map(
                    lambda i: weights[
                        spans[0][i] - self.degree[0] : spans[0][i] + 1,
                        spans[1][i] - self.degree[1] : spans[1][i] + 1,
                    ],
                    np.arange(coords.shape[1]),
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

    def normal(self, p: int, coords: np.ndarray):
        """
        Calculate the outward normal vector at given knots.

        This currently is rather hard coded and not garenteed to generalize to
        other NURBS geometries.

        Parameters
        ----------
        p: int
            Index of patch.
        coords: numpy.ndarray
            Parametric coordinates to evaluate the normals. ``coords``
            should be an array of shape ``(2, n)`` where ``n`` is the number of
            coordinates to evaluate. All coordinates should exits on the boundary
            of the patch.

        Returns
        -------
        points: numpy.ndarray
            Locations in physical space of normal vectors.
        normals: numpy.ndarray
            Normal vectors of unit length.
        """
        assert coords.ndim == 2 and coords.shape[0] == 2

        # Get indices corresponding to each curve based on zero location
        cv_idxs = np.argwhere((coords[0, :] == 0) | (coords[0, :] == 1)).flatten()
        cu_idxs = np.argwhere((coords[1, :] == 0) | (coords[1, :] == 1)).flatten()

        # Arrays to fill
        points = np.zeros(coords.shape)
        normals = np.zeros(coords.shape)

        # Cases when u=0
        for i in cv_idxs:
            dv = np.array(
                self.patches[p].derivatives(coords[0, i], coords[1, i], order=1)
            )

            # Save point in (x, y)
            points[:, i] = dv[0, 0, :-1]

            # Compute normal
            normal = dv[0, 1, :-1][::-1]
            if (normal != 0).any():
                normal[0] *= -1
                normal /= (-1 if coords[0, i] == 0 else 1) * np.sqrt(
                    np.sum(normal**2)
                )

            # Save normal vector
            normals[:, i] = normal

        # Cases when v=0
        for i in cu_idxs:
            du = np.array(
                self.patches[p].derivatives(coords[0, i], coords[1, i], order=1)
            )

            # Save point in (x, y)
            points[:, i] = du[0, 0, :-1]

            # Compute normal
            normal = du[1, 0, :-1][::-1]
            if (normal != 0).any():
                normal[0] *= -1
                normal /= (-1 if coords[1, i] == 1 else 1) * np.sqrt(
                    np.sum(normal**2)
                )

            # Save normal vector
            normals[:, i] = normal

        return points, normals

    def jacobian(self, p: int, coords: np.ndarray):
        """
        Computes the Jacobian matrix :math:`\\frac{\\partial(x, y)}{\\partial(\\hat{x},
        \\hat{y})}` for the pull back from the parametric domain to the physical.

        Parameters
        ----------
        p: int
            Index of patch.
        coords: numpy.ndarray
            Parametric coordinates to evaluate the Jacobian at. ``coords``
            should be an array of shape ``(2, n)`` where ``n`` is the number of
            coordinates to evaluate.

        Returns
        -------
        jacobian: numpy.ndarray
            Jacobian matrix for each set of parametric coordinates or shape
            ``(2, 2, n)``.
        """
        assert coords.ndim == 2 and coords.shape[0] == 2

        # Array to fill
        jacobian = np.zeros((2, 2, coords.shape[-1]))

        # Calculate jacobian
        for i in range(coords.shape[-1]):
            # Compute derivatives
            ders = self.patches[p].derivatives(coords[0, i], coords[1, i], order=1)

            # Fill jacobian
            jacobian[0, :, i] = ders[1][0][:2]
            jacobian[1, :, i] = ders[0][1][:2]

        return jacobian

    def inverse_map(
        self,
        physical_coords: np.ndarray,
        tol: float = 1e-8,
        p: int = 0,
        method=None,
    ):
        """
        Map coordinates in the physical domain to the parametric domain.

        .. note::
           This is only implemented for a single patch.


        Parameters
        ----------
        physical_coords: numpy.ndarray
            Array or coordinates of shape ``(2, n)`` where ``n`` is the
            number of coordinates. ``physical_coords[0, :]`` are the ``x``
            positions and ``physical_coords[1, :]`` are the ``y`` positions.
        tol: float, default=1e-8
            Tolerance of inverse map computed with
            ``scipy.optimize.minimize()``.
        p: int, default=0
            Patch index.
        method: str or callable, default=None
            Type of optimizer passed to ``scipy.optimize.minimize()``.

        Returns
        -------
        coords: numpy.ndarray
            Physical coordinates of shape ``(2, n)``.
        """
        assert physical_coords.ndim == 2 and physical_coords.shape[0] == 2

        def minimize(physical_coord):
            return scipy.optimize.minimize(
                lambda coord: np.sum(
                    (
                        np.array(self.patches[p].evaluate_single((coord)))[:2]
                        - physical_coord
                    )
                    ** 2
                ),
                0.5 * np.ones(2),
                method=method,
                bounds=((0, 1), (0, 1)),
                tol=tol,
            ).x

        # Use scipy.optimize
        return np.array(list(map(minimize, physical_coords.T))).T

    # ========================================================================
    # Plotters

    def plot(
        self,
        num_nodes: int = 256,
        plot_ctrlpts: bool = True,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
    ):
        """
        Create 3-D plot of mesh.

        Parameters
        num_nodes: int, default=256
            Number of positions to sample the mesh for each patch.
        plot_ctrlpts: bool, default=True
            Whether to plot the control points.
        backend: "matplotlib" or "plotly", default="matplotlib"
            Which plotting backend to use.

        Returns
        -------
        fig: matplotlib.axes.Axes or plotly.graph_objects.Figure
            Matplotlib axis of Plotly figure depending on the backend.
        """
        # Get parametric sample locations
        X, Y = np.meshgrid(np.linspace(0, 1, num_nodes), np.linspace(0, 1, num_nodes))
        X = X[..., np.newaxis]
        Y = Y[..., np.newaxis]

        if backend == "matplotlib":
            # Create axis
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})

            # Set of colors for unique materials
            colors = list(mcolors.TABLEAU_COLORS.values())
            mats = {}

            # Iterate through patches
            for patch in self.patches:
                # Sample points
                points = np.array(
                    patch.evaluate_list(
                        np.concatenate([X, Y], axis=2).reshape((-1, 2)).tolist()
                    )
                ).reshape((*X.shape[:-1], 3))

                # Plot patch geometry
                if patch.name in mats:
                    ax.plot_surface(
                        points[..., 0],
                        points[..., 1],
                        points[..., 2],
                        color=mats[patch.name],
                    )
                else:
                    mats[patch.name] = colors.pop(0)
                    ax.plot_surface(
                        points[..., 0],
                        points[..., 1],
                        points[..., 2],
                        color=mats[patch.name],
                        label=patch.name,
                    )

                # Plot control points
                if plot_ctrlpts:
                    ctrlpts = np.array(patch.ctrlpts)
                    ax.scatter(ctrlpts[:, 0], ctrlpts[:, 1], ctrlpts[:, 2], color="k")

            # Label axes
            ax.set_xlabel("$x~(cm)$")
            ax.set_ylabel("$y~(cm)$")
            return ax

        elif backend == "plotly":
            # Iterate through patches
            surfaces = []
            scatter = []
            for p, patch in enumerate(self.patches):
                # Sample points
                points = np.array(
                    patch.evaluate_list(
                        np.concatenate([X, Y], axis=2).reshape((-1, 2)).tolist()
                    )
                ).reshape((*X.shape[:-1], 3))

                # Plot patch geometry
                surfaces.append(
                    go.Surface(
                        x=points[..., 0],
                        y=points[..., 1],
                        z=points[..., 2],
                        coloraxis="coloraxis",
                        name="Surface (Patch: {}, Material: {})".format(p, patch.name),
                    )
                )

                # Plot control points
                if plot_ctrlpts:
                    ctrlpts = np.array(patch.ctrlpts)
                    scatter.append(
                        go.Scatter3d(
                            x=ctrlpts[:, 0],
                            y=ctrlpts[:, 1],
                            z=ctrlpts[:, 2],
                            mode="markers",
                            marker={"color": "black", "size": 5},
                            name="Control Points (Patch: {}, Material: {})".format(
                                p, patch.name
                            ),
                        )
                    )

            # Create figure
            fig = go.Figure(
                data=surfaces + scatter,
            )
            fig.update_layout(
                scene={
                    "xaxis_title": "x (cm)",
                    "yaxis_title": "y (cm)",
                },
                coloraxis={"colorscale": "thermal"},
            )
            return fig

    def plot_normals(self, p, num_nodes=256):
        """
        Plot normals at all boundaries of a given patch.

        Parameters
        ----------
        p: int
            Patch index.
        num_nodes: int
            Number of positions to sample the mesh for each patch.

        Returns
        -------
        ax: matplotlib.axes.Axes
            Resulting matplotlib axis object.
        """
        # Get patch
        patch = self.patches[p]

        # Create axis
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Get parametric sample locations
        X, Y = np.meshgrid(np.linspace(0, 1, num_nodes), np.linspace(0, 1, num_nodes))
        X = X[..., np.newaxis]
        Y = Y[..., np.newaxis]

        # Sample points
        points = np.array(
            patch.evaluate_list(
                np.concatenate([X, Y], axis=2).reshape((-1, 2)).tolist()
            )
        ).reshape((*X.shape[:-1], 3))

        # Plot surface
        ax.plot_surface(
            points[..., 0],
            points[..., 1],
            points[..., 2],
        )

        # Calculate parametric coordinates
        coords = np.zeros((2, 40))
        coords[0, 0:10] = np.linspace(0, 1, 12)[1:-1]
        coords[0, 10:20] = np.linspace(0, 1, 12)[1:-1]
        coords[1, 10:20] = 1
        coords[1, 20:30] = np.linspace(0, 1, 12)[1:-1]
        coords[1, 30:40] = np.linspace(0, 1, 12)[1:-1]
        coords[0, 30:40] = 1

        # Set of colors
        colors = list(mcolors.TABLEAU_COLORS.values())

        # Evaluate normals
        points, normals = self.normal(p, coords)

        labels = ["$\\hat y = 0$", "$\\hat y = 1$", "$\\hat x = 0$", "$\\hat x = 1$"]
        for i, label in enumerate(labels):
            ax.quiver(
                points[0, i * 10 : i * 10 + 10],
                points[1, i * 10 : i * 10 + 10],
                np.zeros(10),
                normals[0, i * 10 : i * 10 + 10],
                normals[1, i * 10 : i * 10 + 10],
                np.zeros(10),
                label=label,
                color=colors[i],
                length=0.1,
            )

        # Label axes
        ax.set_xlabel("$x~(cm)$")
        ax.set_ylabel("$y~(cm)$")
        ax.set_aspect("equal")
        return ax

    # ========================================================================
    # Getters/Setters

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
        phi = phi.reshape((phi.shape[0], -1))
        assert phi.shape[0] == self.num_patches

        # Set control points for scalar field
        for p in range(self.num_patches):
            patch = self.patches[p]
            new_ctrlpts = patch.ctrlpts
            for i in range(phi.shape[-1]):
                new_ctrlpts[i] = [*new_ctrlpts[i][:-1], phi[p, i]]
            patch.ctrlpts = new_ctrlpts
            self.patches[p] = patch

    @property
    def num_patches(self):
        return len(self.patches)

    @property
    def degree(self):
        return [self.patches[0].degree_u, self.patches[0].degree_v]

    @property
    def ndim(self):
        return 2
