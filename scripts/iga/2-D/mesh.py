import itertools
from typing import List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
from boundary import IncidentBoundary
from geomdl import NURBS
from geomdl.helpers import basis_functions, basis_functions_ders, find_spans
from igakit import cad


class IGAMesh(object):
    def __init__(self, patches: dict):
        """"""
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

    def finalize_patches(self):
        """"""
        assert not self._finalized and not self._connected

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
        self._finalized = True

    # ========================================================================
    # Boundary condition methods

    def connect(self, tol=1e-8):
        """"""
        assert self._finalized and not self._connected

        connections = {}
        for p in range(self.num_patches):
            # Get patch
            patch = self.patches[p]

            # Evaluate a point at each surface
            connections[p] = {
                0: patch.evaluate_single((0.5, 0)),
                1: patch.evaluate_single((0.5, 1)),
                2: patch.evaluate_single((0, 0.5)),
                3: patch.evaluate_single((1, 0.5)),
            }

        # Iterate through each interface
        for p_out, b_out in itertools.product(
            np.arange(self.num_patches), np.arange(4)
        ):
            # Check if we should ignore
            if self._incident_boundaries[p_out][b_out] is not None:
                continue

            # Get coordinate
            coord = connections[p_out][b_out]

            for p_in, b_in in itertools.product(
                np.arange(self.num_patches), np.arange(4)
            ):
                if (coord - connections[p_in][b_in] < tol).all():
                    # Boundary of patches match
                    self._incident_boundaries[p_out][b_out] = IncidentBoundary(
                        from_patch=p_in, orientation=1
                    )
                    self._incident_boundaries[p_in][b_in] = IncidentBoundary(
                        from_patch=p_out, orientation=1
                    )

            assert self._incident_boundaries[p_out][b_out] is not None

        self._connected = True

    def set_boundary_condition(
        self,
        p: int,
        const_param_idx: int,
        const_param: int,
        condition: Literal["vacuum", "reflective"],
    ):
        """"""
        assert self._finalized and not self._connected

        # Set boundary condition
        self._incident_boundaries[p][
            2 * abs(const_param_idx - 1) + const_param
        ] = IncidentBoundary(
            from_patch=None if condition == "vacuum" else p,
            orientation=self.get_orientation(p, const_param_idx, const_param),
        )

    def get_boundary(self, p: int, const_param_idx: int, const_param: int):
        """"""
        return self._incident_boundaries[p][2 * abs(const_param_idx - 1) + const_param]

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
        """"""
        assert not self._finalized

        # Refine patch
        self.patches[p] = cad.refine(self.patches[p], factor=factor, degree=degree)

    # ========================================================================
    # Parametric methods

    def find_spans(self, patch_idx: int, param_idx, knots):
        """
        Compute the knot spans along a parametric dimension for a vector of knots.

        Parameters
        ----------
        patch_idx: int
            Index of patch in ``patches``.
        knots: numpy.ndarray
            Locations to find the knot span. ``knots`` should be an array of shape
            ``(K,)`` where ``K`` is the number of knots to evaluate.

        Returns
        -------
        spans: np.ndarray
            Index in the knotvector for the start of the knot span that
            the given knot is located in.
        """
        knots = np.array(knots, dtype=float)
        assert knots.ndim == 1

        # Find knot spans from knotvector
        return np.array(
            find_spans(
                self.degree[param_idx],
                self.patches[patch_idx].knotvector[param_idx],
                self.patches[patch_idx].ctrlpts_size,
                knots,
            )
        )

    def basis_functions(self, patch_idx: int, knots: np.ndarray):
        """
        Compute all non-vanishing 2-D NURBS basis functions. The current implementation
        assumes an open knot vector with single multiplicity on inner knots.

        Parameters
        ----------
        patch_idx: int
            Index of patch in ``patches``.
        knots: numpy.ndarray
            Locations to evaluate the basis functions. ``knots`` should be an array of
            shape ``(2, K)`` where ``K`` is the number of knots to evaluate.

        Returns
        -------
        basis_data: numpy.ndarray
            Non-zero basis function evaluations of shape ``(K, p + 1, q + 1)`` where
            ``p`` and ``q`` are the degrees of the basis functions along each respective
            parametric dimension.
        """
        assert knots.ndim == 2 and knots.shape[0] == 2

        # Get weights
        weights = np.array(self.patches[patch_idx].weights).reshape(
            (
                self.patches[patch_idx].ctrlpts_size_u,
                self.patches[patch_idx].ctrlpts_size_v,
            )
        )

        # Compute B-Spline basis
        spans = [[] for _ in range(self.ndim)]
        basis = [[] for _ in range(self.ndim)]

        for i in range(self.ndim):
            # Handle cases when we're evaluating at exactly 1
            spans[i] = (
                len(self.patches[patch_idx].knotvector[i]) - 2 * self.degree[i]
            ) * np.ones(knots.shape[1], dtype=int)
            basis[i] = np.zeros((knots.shape[1], self.degree[i] + 1))
            basis[i][:, -1] = 1

            # Calculate cases when the coordinate < 1
            if (knots[i, :] != 1).any():
                spans[i][knots[i, :] != 1] = np.array(
                    self.find_spans(patch_idx, i, knots[i, :][knots[i, :] != 1])
                )
                basis[i][knots[i, :] != 1, :] = np.array(
                    basis_functions(
                        self.degree[i],
                        self.patches[patch_idx].knotvector[i],
                        spans[i][knots[i, :] != 1],
                        knots[i, :][knots[i, :] != 1],
                    )
                )

        # Evaluate numerator of 2-D NURBS basis
        basis_data = np.einsum("ij,ik->ijk", basis[0], basis[1]) * np.array(
            list(
                map(
                    lambda i: weights[
                        spans[0][i] - self.degree[0] : spans[0][i] + 1,
                        spans[1][i] - self.degree[1] : spans[1][i] + 1,
                    ],
                    np.arange(knots.shape[1]),
                ),
            )
        )

        # Evaluate denominator and return
        return basis_data / np.sum(np.sum(basis_data, axis=-1), axis=-1).reshape(
            (-1, 1, 1)
        )

    def basis_function_grads(self, patch_idx: int, knots: np.ndarray, order: int = 1):
        """
        Compute gradients of the basis functions for a given patch. This implementation
        assumes open knot vectors with all interior knots of single multiplicity.

        Parameters
        ----------
        patch_idx: int
            Index of patch in ``patches``.
        knots: numpy.ndarray
            Locations to evaluate the gradient. ``knots`` should be an array of shape
            ``(2, K)`` where ``K`` is the number of knots to evaluate.
        order: int, default=1
            Derivative order to be evaluated.

        Returns
        -------
        basis_data: numpy.ndarray
            Basis data of shape ``(K, p + 1, q + 1, 3)`` where ``p`` and ``q`` are the
            degrees of the basis functions for each parametric dimension.
            ``(:, :, :, 0)``
            contains non-vanishing basis function evaluations, ``(:, :, :, 1)`` contains
            non-vanishing basis function derivatives with respect to ``u`` and
            ``(:, :, :, 2)`` contains non-vanishing basis function derivatives with
            respect to ``v``.
        """
        assert knots.ndim == 2 and knots.shape[0] == 2

        # Get weights
        weights = np.array(self.patches[patch_idx].weights).reshape(
            (
                self.patches[patch_idx].ctrlpts_size_u,
                self.patches[patch_idx].ctrlpts_size_v,
            )
        )

        # Compute B-Spline basis
        spans = [[] for _ in range(self.ndim)]
        basis = [[] for _ in range(self.ndim)]
        basis_ders = [[] for _ in range(self.ndim)]

        for i in range(self.ndim):
            spans[i] = np.array(self.find_spans(patch_idx, i, knots[i, :]))
            evals = np.array(
                basis_functions_ders(
                    self.degree[i],
                    self.patches[patch_idx].knotvector[i],
                    spans[i],
                    knots[i, :],
                    order,
                )
            )

            basis[i] = evals[:, 0, :]
            basis_ders[i] = evals[:, 1, :]

        basis_data = np.zeros(
            (knots.shape[1], self.degree[0] + 1, self.degree[1] + 1, 3)
        )

        # Compute numerator of basis functions
        weights = np.array(
            list(
                map(
                    lambda i: weights[
                        spans[0][i] - self.degree[0] : spans[0][i] + 1,
                        spans[1][i] - self.degree[1] : spans[1][i] + 1,
                    ],
                    np.arange(knots.shape[1]),
                ),
            )
        )
        basis_data[..., 0] = np.einsum("ij,ik->ijk", basis[0], basis[1]) * weights

        # Compute denominator
        denom = np.sum(np.sum(basis_data[..., 0], axis=-1), axis=-1).reshape((-1, 1, 1))

        # Scale basis functions by denominator
        basis_data[..., 0] /= denom

        # Compute first term of quotient rule
        basis_data[..., 1] += (
            np.einsum("ij,ik->ijk", basis_ders[0], basis[1]) * weights / denom
        )
        basis_data[..., 2] += (
            np.einsum("ij,ik->ijk", basis[0], basis_ders[1]) * weights / denom
        )

        # Compute second term of quotient rule
        basis_data[..., 1] -= basis_data[..., 0] * np.sum(
            np.sum(basis_data[..., 1], axis=-1), axis=-1
        ).reshape((-1, 1, 1))
        basis_data[..., 2] -= basis_data[..., 0] * np.sum(
            np.sum(basis_data[..., 2], axis=-1), axis=-1
        ).reshape((-1, 1, 1))

        return basis_data

    def normal(self, patch_idx: int, knots: np.ndarray):
        """
        Calculate the outward normal vector at given knots.

        This currently is rather hard coded and not garenteed to generalize to
        other NURBS geometries.

        Parameters
        ----------
        patch_idx: int
            Index of patch in ``patches``.
        knots: numpy.ndarray
            Locations to evaluate the normal vector (must be along the boundary).
            ``knots`` should be an array of shape ``(2, K)`` where ``K`` is the
            number of knots to evaluate.

        Returns
        -------
        points: numpy.ndarray
            Locations in physical space of normal vectors.
        normals: numpy.ndarray
            Normal vectors of unit length.
        """
        assert knots.ndim == 2 and knots.shape[0] == 2

        # Get indices corresponding to each curve based on zero location
        cv_idxs = np.argwhere((knots[0, :] == 0) | (knots[0, :] == 1)).flatten()
        cu_idxs = np.argwhere((knots[1, :] == 0) | (knots[1, :] == 1)).flatten()

        # Arrays to fill
        points = np.zeros(knots.shape)
        normals = np.zeros(knots.shape)

        # TODO: Currently I'm flipping when -1 gets applied based on the directional
        # derivative. This ensures the normal points outward for the circle in pu.py
        # but is not necessarily general to other NURBS surfaces.

        # Cases when u=0
        for i in cv_idxs:
            dv = np.array(
                self.patches[patch_idx].derivatives(knots[0, i], knots[1, i], order=1)
            )

            # Save point in (x, y)
            points[:, i] = dv[0, 0, :-1]

            # Compute normal
            normal = dv[0, 1, :-1][::-1]
            if (normal != 0).any():
                normal[0] *= -1
                normal /= (-1 if knots[0, i] == 0 else 1) * np.sqrt(np.sum(normal**2))

            # Save normal vector
            normals[:, i] = normal

        # Cases when v=0
        for i in cu_idxs:
            du = np.array(
                self.patches[patch_idx].derivatives(knots[0, i], knots[1, i], order=1)
            )

            # Save point in (x, y)
            points[:, i] = du[0, 0, :-1]

            # Compute normal
            normal = du[1, 0, :-1][::-1]
            if (normal != 0).any():
                normal[0] *= -1
                normal /= (-1 if knots[1, i] == 1 else 1) * np.sqrt(np.sum(normal**2))

            # Save normal vector
            normals[:, i] = normal

        return points, normals

    def jacobian(self, patch_idx: int, knots: np.ndarray):
        """
        Computes the Jacobian matrix :math:`\\frac{\\partial(x, y)}{\\partial(\\hat{x},
        \\hat{y})}` for the pull back from the parametric domain to the physical.

        Parameters
        ----------
        patch_idx: int
            Index of patch in ``patches``.
        knots: numpy.ndarray
            Locations to evaluate the Jacobian matrix.
            ``knots`` should be an array of shape ``(2, K)`` where ``K`` is the
            number of knots to evaluate.

        Returns
        -------
        jacobian: numpy.ndarray
            Jacobian matrix for each set of parametric coordinates.
        """
        assert knots.ndim == 2 and knots.shape[0] == 2

        # Array to fill
        jacobian = np.zeros((2, 2, knots.shape[-1]))

        # Calculate jacobian
        for i in range(knots.shape[-1]):
            # Compute derivatives
            ders = self.patches[patch_idx].derivatives(
                knots[0, i], knots[1, i], order=1
            )

            # Fill jacobian
            jacobian[0, :, i] = ders[1][0][:2]
            jacobian[1, :, i] = ders[0][1][:2]

        return jacobian

    def inverse_map(
        self,
        physical_coords: np.ndarray,
        tol: float = 1e-8,
        method=None,
    ):
        """"""
        assert physical_coords.ndim == 2 and physical_coords.shape[0] == 2

        # TODO: add multipatch
        p = 0

        # Create initial guess
        coords = 0.5 * np.ones(physical_coords.shape)

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
        coords = np.array(list(map(minimize, physical_coords.T))).T

        return coords

    # ========================================================================
    # Plotters

    def plot(self, num_nodes=256, plot_ctrlpts=True):
        # Create axis
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Get parametric sample locations
        X, Y = np.meshgrid(np.linspace(0, 1, num_nodes), np.linspace(0, 1, num_nodes))
        X = X[..., np.newaxis]
        Y = Y[..., np.newaxis]

        # Sample points
        points = np.array(
            self.patches[0].evaluate_list(
                np.concatenate([X, Y], axis=2).reshape((-1, 2)).tolist()
            )
        ).reshape((*X.shape[:-1], 3))

        # Plot patch
        ax.plot_surface(points[..., 0], points[..., 1], points[..., 2])
        ax.set_xlabel("$x~(cm)$")
        ax.set_ylabel("$y~(cm)$")

        # Plot control points
        if plot_ctrlpts:
            ctrlpts = np.array(self.patches[0].ctrlpts)
            ax.scatter(ctrlpts[:, 0], ctrlpts[:, 1], ctrlpts[:, 2], color="k")
        return ax

    # ========================================================================
    # Getters/Setters

    def num_dofs(self, all=False):
        return (
            self.patches[0].ctrlpts_size
            if not all
            else self.num_patches * self.patches[0].ctrlpts_size
        )

    @property
    def num_patches(self):
        return len(self.patches)

    @property
    def degree(self):
        return [self.patches[0].degree_u, self.patches[0].degree_v]

    @property
    def ndim(self):
        return 2

    @property
    def num_dofs_axis(self):
        return (self.patches[0].ctrlpts_size_u, self.patches[0].ctrlpts_size_v)

    def set_phi(self, p, phi):
        """"""
        patch = self.patches[p]
        new_ctrlpts = patch.ctrlpts
        for i in range(phi.shape[-1]):
            new_ctrlpts[i] = [*new_ctrlpts[i][:-1], phi[i]]
        patch.ctrlpts = new_ctrlpts
        self.patches[p] = patch
