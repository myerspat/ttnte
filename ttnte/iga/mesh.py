import itertools
import multiprocessing as mp
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import cotengra as ctg
import h5py as h5
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch as tn
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ttnte.cad.patch import Patch


class IGAMesh(object):
    """
    IGA mesh object. Handles boundary information for connecting
    patches, plotting, etc.

    Attributes
    ----------
    num_patches: int
        Number of patches in the mesh
    patches: dict of int and ttnte.cad.Patch
        The patches in the mesh.
    patch_ids: list of int
        A list of all the patch IDs in the mesh.
    max_processes: int
        Max number of processes at once.
    name: str or None
        Name of mesh object.
    id: int
        ID of mesh object.
    has_reflective_boundary: bool
        The mesh has a reflective boundary.
    """

    # IGAMesh ID iterator
    _id_iter = itertools.count()

    def __init__(self, **kwargs):
        """
        Initialize IGAMesh object.

        .. warning::
           There must be no hanging nodes between patches each
           defined with open knot vectors and internal knot multiplicity
           of one.

        Parameters
        ----------
        name: str or None
            Name of the mesh.
        max_processes: int
            Maximum number of processes allowed. Defaults to
            ``multiprocessing.cpu_count() - 1``.
        """
        # Initialize patches
        self._patches = {}

        # Unpack kwargs
        self._name = kwargs.get("name", None)
        self._max_processes = kwargs.get("max_processes", max(1, mp.cpu_count() - 1))

        # States
        self._bcs_set = False
        self._finalized = False
        self._connected = False
        self._mapped_regular_mesh = False
        self._has_reflective_boundary = False

        # Set the ID
        self._id = next(self._id_iter)

        # Other variables
        self._decimals = 8
        self._boundary_hash = {}
        self._bboxes = []

    # ========================================================================
    # Methods for modifying the mesh

    def add_patch(self, patch: Patch):
        """
        Add patch to the mesh.

        Parameters
        ----------
        patch: ttnte.cad.Patch
            New patch.
        """
        assert not self._connected

        # Throw warning if patch has already been added
        if patch.id in self._patches:
            warnings.warn(f"Already added {patch} to IGAMesh")
        else:
            # Add patch
            self._patches[patch.id] = patch

    def refine(
        self,
        factor: Union[int, List[int]],
        degree: Union[int, List[int]],
    ):
        """
        Refine all the patches in the mesh.

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
        assert not self._finalized

        for patch in self._patches.values():
            patch.refine(factor, degree)

    def connect(self, decimals=8):
        """
        Connect patches. boundary conditions. Patches not connected are assumed to be
        vacuum boundary conditions unless Determine patch interfaces for passing
        otherwise specified in :meth:`ttnte.iga.IGAMesh.set_reflective_condition`.

        Convert patches and connect them. Patch boundaries not connected to another
        patch are assumed to have vacuum boundaries unless specified in
        :meth:`tttnte.iga.IGAMesh.set_reflective_conditions`

        Parameters
        ----------
        decimals: int, default=8
            Number of decimals to round boundary evaluations. These
            are matched with other patch boundaries to determine
            which share boundaries.
        """
        assert not self._finalized and not self._connected
        self._decimals = decimals

        # Find all boundaries and build spatial hash table
        self._boundary_hash = {}
        self._bboxes = []

        # Convert and initialize all patches
        for patch in self._patches.values():
            # Convert patch to geomdl backend
            patch.igakit2geomdl()

            # Get bounding box
            self._bboxes.append(np.array(patch.bbox)[:, :-1])

            # Initialize patch boundaries
            boundaries = patch.initialize_boundaries(self._decimals)

            # Add new boundaries or append for existing
            for point, pid in boundaries.items():
                if point in self._boundary_hash:
                    self._boundary_hash[point].append(pid)
                else:
                    self._boundary_hash[point] = [pid]

        # Check connections
        for connections in self._boundary_hash.values():
            if len(connections) > 2 and not np.all(np.array(connections) == None):
                raise RuntimeError(
                    "The boundary of each patch should have at most one other neighbor"
                )

        self._connected = True

    def set_reflective_conditions(
        self,
        faces: Union[
            tuple,
            Literal["left", "right", "bottom", "top"],
        ],
    ):
        """
        Set the reflective boundary conditions.

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
                if self._boundary_hash[tuple(boundaries[i, :])][0] is not None:
                    self._boundary_hash[tuple(boundaries[i, :])] += self._boundary_hash[
                        tuple(boundaries[i, :])
                    ]

        self._has_reflective_boundary = True

    def finalize(self):
        """Finalize mesh."""
        self._finalized = True

    # ========================================================================
    # Boundary condition methods

    def get_connected_patch(self, pid: int, centroid: tuple):
        """
        Get the patch a specific patch's boundary connects to.

        .. warning::
           There must be no hanging nodes between patches;
           therefore, exactly one boundary of the given patch
           connects to another.

        Parameters
        ----------
        p: int
            Index of patch.
        coord: (0, 0.5), (1, 0.5), (0.5, 0), or (0.5, 1)
            Parametric coordinate of center of boundary.

        Returns
        -------
        connected_p: int or None
            Index of the connected patch. If it is ``None`` then this
            boundary is a vacuum boundary condition.
        """
        if not self._connected or not self._finalized:
            raise RuntimeError("Patches must be connected and finalized")

        # Calculate physical point
        point = tuple(
            np.round(np.array(self._patches[pid](centroid)[:-1]), self._decimals)
        )

        # Get connected patch ID
        pids = np.array(self._boundary_hash[point])

        if pids[0] == None or len(pids) == 1:
            # Vacuum or no boundary
            return None
        elif (pids != pid).any():
            # Transmission boundary
            return pids[pids != pid][0]
        else:
            # Reflective boundary
            return pid

    def check_boundary_existance(self, pid: int, centroid: tuple):
        """
        Check if a boundary exists.

        Parameters
        ----------
        p: int
            Index of patch.
        coord: (0, 0.5), (1, 0.5), (0.5, 0), or (0.5, 1)
            Parametric coordinate of center of boundary.

        Returns
        -------
        exists: bool
            Whether the boundary has a non-zero length in physical space.
        """
        if not self._connected or not self._finalized:
            raise RuntimeError("Patches must be connected and finalized")

        # Calculate physical point
        point = tuple(
            np.round(np.array(self._patches[pid](centroid)[:-1]), self._decimals)
        )

        # Get connected patch ID
        pids = np.array(self._boundary_hash[point])
        return pids[0] != None

    # ========================================================================
    # Mapping methods

    def inverse_map(
        self,
        physical_coords: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8,
        batch_size: int = 500,
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
        batch_size: int, default=500
            Number of samples to pass to ``ttnte.cad.Patch.inverse_map()``
            for each process.

        Returns
        -------
        coords: numpy.ndarray
            Physical coordinates of shape ``(n, 2)``.
        """
        assert physical_coords.ndim == 2 and physical_coords.shape[1] == 2

        # Find candidate patches
        pid_iterators = []
        if self.num_patches > 1:
            for i in range(physical_coords.shape[0]):
                pid_iterators.append([])
                for pid, bbox in zip(self._patches.keys(), self._bboxes):
                    if (bbox[0, :] - 0.00005 <= physical_coords[i, :]).all() and (
                        bbox[1, :] + 0.00005 >= physical_coords[i, :]
                    ).all():
                        pid_iterators[-1].append(pid)

        else:
            pid_iterators = [
                [list(self._patches.keys())[0]] for _ in range(physical_coords.shape[0])
            ]

        # Convert to np.array of iterators
        pid_iterators = np.array([iter(it) for it in pid_iterators])

        # Array for coordinates
        coords = np.zeros((physical_coords.shape[0], 2))
        pids = np.zeros(physical_coords.shape[0], dtype=int)

        # Distances
        old_distances = np.ones(physical_coords.shape[0])
        new_distances = old_distances.copy()

        # Evaluate coarse mesh on all patches
        X, Y = np.meshgrid(np.linspace(0, 1, 12)[1:-1], np.linspace(0, 1, 12)[1:-1])
        points = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=-1)
        mesh = np.array(self(points))[:, :, :-1]

        while True:
            # Get indices of the unconverged
            unconverged = new_distances >= tol

            # Check if converged
            if (unconverged == False).all():
                return pids, coords

            # Get next candidate patches
            pids[unconverged] = np.array(
                [next(it, -1) for it in pid_iterators[unconverged]]
            )
            unique_pids = np.unique(pids[unconverged])

            # Check if candidate patches have been exhausted
            if (unique_pids == -1).any():
                raise RuntimeError("Failed to converge on all candidate patches")

            # Find closest starting points
            idxs = np.array(
                [np.argwhere(unique_pids == pid) for pid in pids[unconverged]]
            ).flatten()
            local_distances = np.linalg.norm(
                (
                    mesh[idxs, :, :]
                    - ctg.einsum(
                        "a,bc->bac",
                        np.ones((points.shape[0])),
                        physical_coords[unconverged, :],
                    )
                ).reshape((-1, 2)),
                2,
                axis=-1,
            ).reshape((idxs.size, mesh.shape[1]))

            # Get minimum distances and set corresponding points
            coords[unconverged, :] = points[np.argmin(local_distances, axis=-1), :]
            old_distances[unconverged] = np.min(local_distances, axis=-1)

            # Run all unique ids
            with mp.Pool(self._max_processes) as pool:
                # Create a mask for each unique ID
                masks = [unconverged & (pids == pid) for pid in unique_pids]

                # Build arg set
                args = []
                for pid, mask in zip(unique_pids, masks):
                    # Get the number of batches
                    num_batches = np.ceil(np.sum(mask) / batch_size).astype(int)

                    # Fill arguments
                    args += [
                        (
                            pid,
                            physical_coords[mask, :][
                                i * batch_size : i * batch_size + batch_size, :
                            ],
                            max_iter,
                            tol,
                            coords[mask, :][
                                i * batch_size : i * batch_size + batch_size, :
                            ],
                        )
                        for i in range(num_batches)
                    ]

                # Run pool
                coords_list = pool.starmap(self._inverse_map_patch, args)

                # Update coordinates
                i = 0
                for pid, mask in zip(unique_pids, masks):
                    # Get the number of batches
                    num_batches = np.ceil(np.sum(mask) / batch_size).astype(int)

                    # Concatenate the arrays
                    coords[mask, :] = np.concatenate(
                        coords_list[i : i + num_batches], axis=0
                    )
                    i += num_batches

                    # Calculate new distances
                    new_distances[mask] = np.linalg.norm(
                        self._patches[pid](coords[mask, :])[:, :-1]
                        - physical_coords[mask, :],
                        2,
                        axis=-1,
                    )

                pool.close()

            # Update old distances
            old_distances[unconverged] = new_distances[unconverged]

    def _inverse_map_patch(self, pid, physical_coords, max_iter, tol, coords):
        return self._patches[pid].inverse_map(physical_coords, max_iter, tol, coords)[0]

    def map_regular_mesh(
        self,
        shape: Tuple[int] = (128, 128),
        N: Tuple[int] = (5, 5),
        max_iter: int = 100,
        tol: float = 1e-8,
        batch_size: int = 500,
    ):
        """
        Find patches and parametric coordinates for cell averaging the scalar flux to a
        regular mesh.

        .. warning::
            This function only works for problems with axis-aligned
            boundaries.

        Parameters
        ----------
        shape: tuple of int, default=(128, 128)
            Number of cells along each axis.
        N: tuple of int, default=(5, 5)
            Discretization within each cell for trapezoidal integration.
        max_iter: int, default=100
            Max number of iterations for Newton-Raphson in inverse map.
        tol: float, default=1e-8
        batch_size: int, default=500
            Number of samples in each process of ``ttnte.cad.Patch.inverse_map()``.

        Returns
        -------
        pids: numpy.ndarray
            The patch IDs for each coordinate in the regular mesh. The
            resulting shape is ``(*shape, *N)``.
        coords: numpy.ndarray
            The parametric coordinates for each point. The resulting
            shape is ``(*shape, *N, 2)``.
        """
        assert N[0] > 1 and N[1] > 1

        # Find bounding box for all patches
        full_bbox = np.zeros((2, 2))
        full_bbox[0, :] = np.inf

        for bbox in self._bboxes:
            if (bbox[0, :] < full_bbox[0, :]).all():
                full_bbox[0, :] = bbox[0, :]
            if (bbox[1, :] > full_bbox[1, :]).all():
                full_bbox[1, :] = bbox[1, :]

        # Create regular mesh edges
        x = np.linspace(full_bbox[0, 0], full_bbox[1, 0], shape[0] + 1)
        y = np.linspace(full_bbox[0, 1], full_bbox[1, 1], shape[1] + 1)
        points = np.zeros((*shape, *N, 2))
        for i in range(shape[0]):
            # Get cell bounds
            xl = x[i]
            xr = x[i + 1]

            for j in range(shape[1]):
                # Get cell bounds
                yl = y[j]
                yr = y[j + 1]

                # Get mesh that needs to be evaluated
                X, Y = np.meshgrid(np.linspace(xl, xr, N[0]), np.linspace(yl, yr, N[1]))
                points[i, j, ...] = np.concatenate(
                    [
                        X[..., np.newaxis],
                        Y[..., np.newaxis],
                    ],
                    axis=-1,
                )

        # Apply inverse map
        pids, coords = self.inverse_map(
            points.reshape((-1, 2)), max_iter=max_iter, tol=tol
        )

        # Reshape solution
        pids = pids.reshape((*shape, *N))
        coords = coords.reshape((*shape, *N, 2))

        return pids, coords

    def regular_mesh(
        self,
        pids: np.ndarray,
        coords: np.ndarray,
        dx=1,
        dy=1,
    ):
        """
        Calculate volume averaged scalar flux conforming to a regular mesh.

        .. warning::
            This function only works for problems with axis-aligned
            boundaries.

        Parameters
        ----------
        pids: numpy.ndarray
            The patch IDs for each coordinate in the regular mesh.
        coords: numpy.ndarray
            The parametric coordinates for each point.

        Returns
        -------
        phi: numpy.ndarray
            The volume averaged scalar flux for a regular mesh calculated
            with trapezoidal integration.
        """
        assert (*pids.shape, 2) == coords.shape

        # Evaluate functions
        patch_points_list = self(
            {pid: coords[pids == pid, :] for pid in self._patches.keys()}
        )

        points = np.zeros(pids.shape)
        for pid, patch_points in zip(self.patch_ids, patch_points_list):
            points[pids == pid] = patch_points[..., -1]

        # Compute new averaged solution using trap rule
        return (
            1
            / 4
            * (
                points[..., 0, 0]
                + points[..., -1, 0]
                + points[..., 0, -1]
                + points[..., -1, -1]
            )
            + 1
            / 2
            * (
                np.sum(points[..., 1:-1, 0], axis=-1)
                + np.sum(points[..., 1:-1, -1], axis=-1)
                + np.sum(points[..., 0, 1:-1], axis=-1)
                + np.sum(points[..., 1, 1:-1], axis=-1)
            )
            + np.sum(np.sum(points[..., 1:-1, 1:-1], axis=-1), axis=-1)
        ) / ((coords.shape[-2] - 1) ** 2)

    # ========================================================================
    # Parallel methods

    def _run_across_patches(self, method: Callable, args):
        # Fill in coords
        with mp.Pool(min(self._max_processes, mp.cpu_count() - 1)) as pool:
            result = pool.starmap(method, args)
            pool.close()
            return result

    def _get_arguments(self, coords: Union[np.ndarray, Dict[int, np.ndarray]]):
        return (
            [(self._patches[pid], coords[pid]) for pid in coords.keys()]
            if isinstance(coords, dict)
            else zip(self._patches.values(), itertools.repeat(coords))
        )

    def __call__(self, coords: Union[np.ndarray, Dict[int, np.ndarray]]):
        """
        Evaluate the mesh.

        Parameters
        ----------
        coords: numpy.ndarray or a dict of int and numpy.array
            The parametric coordinates to evaluate the mesh. If a ``numpy.ndarray``
            is given then assume that all patches get the same coordiantes. If
            it is a dictionary then the keys are the patch IDs and the
            arrays are the coordinates to evaluate each patch.

        Returns
        -------
        physical_coords: numpy.ndarray or a dict of int and numpy.array
            The solution after evaluation.
        """
        return self._run_across_patches(Patch.evaluate, self._get_arguments(coords))

    # ========================================================================
    # Plotters

    def plot(
        self,
        num_nodes: int = 256,
        plot_ctrlpts: bool = True,
        use_3d: bool = False,
        color_by: Literal["material", "patch"] = "material",
        cmap: str = "plasma",
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        **kwargs,
    ):
        """
        Create 2-D or 3-D plot of mesh.

        Parameters
        num_nodes: int, default=256
            Number of positions to sample the mesh for each patch.
        plot_ctrlpts: bool, default=True
            Whether to plot the control points.
        use_3d: bool, default=False
            Plot mesh in 2-D or 3-D.
        cmap: str, default="plasma"
            Matplotlib colormap.
        meshlines: bool, default=True
            Add or remove mesh lines.
        figsize: None or tuple of int, default=None
            Figure size. Passed to ``matplotlib.pyplot.subplots()``.
        backend: "matplotlib" or "plotly", default="matplotlib"
            Which plotting backend to use. The plotly implementation
            will only plot in 3-D.

        Returns
        -------
        ax: matplotlib.axes.Axes or plotly.graph_objects.Figure
            Matplotlib axis or Plotly figure depending on the backend.
        """
        # Get parametric sample locations
        X, Y = np.meshgrid(np.linspace(0, 1, num_nodes), np.linspace(0, 1, num_nodes))

        # Calculate points for all surfaces
        points = np.array(
            self(np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1))
        ).reshape((self.num_patches, num_nodes, num_nodes, 3))
        del X, Y

        # Minimum and maximum value
        vmin, vmax = np.min(points[..., -1]), np.max(points[..., -1])

        # Check if z components have been set for control points
        categorical = (
            np.array(list(self._patches.values())[0].ctrlpts)[..., -1] == 0
        ).all()

        # Handle figure size if not given
        figsize = kwargs.pop("figsize", None)

        # Legend updates
        legend_args = {"loc": "upper right", "fancybox": True}
        legend_args.update(kwargs)

        if backend == "matplotlib":
            # Get setttings
            kwargs = {}
            if categorical:
                use_3d = False
                pltcolors = list(mcolors.TABLEAU_COLORS.values())
                color_idxs = {}
                legend_handles = []

                # Iterate through patches
                for i, patch in enumerate(self._patches.values()):
                    # Get label specified by user
                    label = (
                        patch.material
                        if color_by == "material"
                        else (
                            patch.name
                            if patch.name is not None
                            else f"Patch {patch.id}"
                        )
                    )

                    # Check if label exists already
                    if label not in color_idxs:
                        # Add color
                        color_idxs[label] = len(color_idxs)

                        # Add to legend
                        legend_handles.append(
                            mpatches.Patch(
                                color=pltcolors[color_idxs[label]],
                                label=label,
                            )
                        )

                    # Set points
                    points[i, :, :, 2] = color_idxs[label]

                # Set settings
                vmin, vmax = 0, len(list(color_idxs.keys()))
                kwargs["cmap"] = mcolors.ListedColormap(pltcolors[:vmax])

            else:
                # Set settings
                kwargs["cmap"] = cmap

            # Add normalizer
            kwargs["norm"] = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # 2D plot
            if not use_3d:
                # Create axis
                _, ax = plt.subplots(figsize=figsize)

                # 2D settings
                kwargs["shading"] = "gouraud"

                for i, patch in enumerate(self._patches.values()):
                    # Plot 2D surface
                    cmesh = ax.pcolormesh(
                        points[i, ..., 0],
                        points[i, ..., 1],
                        points[i, ..., 2],
                        **kwargs,
                    )

                    # Plot control points
                    if plot_ctrlpts:
                        ctrlpts = np.array(patch.ctrlpts).reshape((-1, 3))
                        sc = ax.scatter(
                            ctrlpts[:, 0],
                            ctrlpts[:, 1],
                            color="k",
                            label="Control Variables" if i == 0 else None,
                        )

                        # Save for legend
                        if i == 0 and categorical:
                            legend_handles.append(sc)

                ax.set_xlabel("$x(\\hat{x}, \\hat{y})~(cm)$")
                ax.set_ylabel("$y(\\hat{x}, \\hat{y})~(cm)$")
                ax.spines[["right", "top"]].set_visible(False)
                ax.set_aspect("equal")
                if plot_ctrlpts:
                    # Extend box outward
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
                    diffx = (abs(xlim[0]) + abs(xlim[1])) * 0.05 / 2
                    diffy = (abs(ylim[0]) + abs(ylim[1])) * 0.05 / 2
                    ax.set_xlim((xlim[0] - diffx, xlim[1] + diffx))
                    ax.set_ylim((ylim[0] - diffy, ylim[1] + diffy))
                if not categorical:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    return ax, plt.colorbar(cmesh, cax=cax)

            else:
                # Create axis
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)

                # 3D settings
                if categorical:
                    kwargs["shade"] = False

                for i, patch in enumerate(self._patches.values()):
                    # Plot 3D surface
                    ax.plot_surface(
                        points[i, ..., 0],
                        points[i, ..., 1],
                        points[i, ..., 2],
                        **kwargs,
                    )

                    # Plot control points
                    if plot_ctrlpts:
                        ctrlpts = np.array(patch.ctrlpts).reshape((-1, 3))
                        sc = ax.scatter(
                            ctrlpts[:, 0],
                            ctrlpts[:, 1],
                            ctrlpts[:, 2],
                            color="k",
                            label="Control Variables" if i == 0 else None,
                        )

                        # Save for legend
                        if i == 0 and categorical:
                            legend_handles.append(sc)

                ax.set_xlabel("$x(\\hat{x}, \\hat{y})~(cm)$")
                ax.set_ylabel("$y(\\hat{x}, \\hat{y})~(cm)$")
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                if not categorical:
                    return ax

            if categorical and (self.num_patches > 1 or plot_ctrlpts):
                ax.legend(handles=legend_handles, **legend_args)
            return ax

        elif backend == "plotly":
            # Iterate through patches
            surfaces = []
            scatter = []
            for i, patch in enumerate(self._patches.values()):
                # Plot patch geometry
                surfaces.append(
                    go.Surface(
                        x=points[i, ..., 0],
                        y=points[i, ..., 1],
                        z=points[i, ..., 2],
                        coloraxis="coloraxis",
                        name="Surface (Patch: {}, Material: {})".format(
                            patch.id, patch.material
                        ),
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
                                patch.id, patch.material
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

    # ========================================================================
    # Overloads

    def __str__(self):
        # Print for terminal
        mstr = (
            f"IGAMesh(id={self._id}, name={self._name}, num_patches={self.num_patches}, "
            + f"reflective_boundaries={self._has_reflective_boundary})\n"
        )
        pstr = "\n".join(f"  -> {patch}" for patch in self._patches.values())

        # Print for terminal
        return mstr + pstr

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        # Print for terminal
        mstr = (
            f"IGAMesh(id={self._id}, name={self._name}, num_patches={self.num_patches}, "
            + f"reflective_boundaries={self._has_reflective_boundary})"
        )
        pstr = "\n".join(
            [f"&nbsp;&nbsp;&nbsp;&nbsp;→ {p}" for p in self._patches.values()]
        )

        # Fancy print for collapsible content
        return (
            f"<details open><summary><code>{mstr}</code>"
            + f"</summary><pre>{pstr}</pre></details>"
        )

    # ========================================================================
    # I / O

    def save(
        self,
        path="mesh.hdf5",
        solution: Optional[np.ndarray] = None,
        k: Optional[float] = None,
    ):
        """
        Save IGA mesh to hdf5 file.

        .. note::
            ``solution`` should be either the angular or scalar flux. For
            the angular flux it should be shape ``(N, P, A, B)`` where
            ``N`` is the number of ordinates, ``P`` is the number of patches,
            ``A`` is the number of control points along ``u``, and ``B``
            is the number of control points along ``v``. The shape of the
            scalar flux should be ``(P, A, B)``.

        Parameters
        ----------
        path: str or PathLike
            Location to save the h5 mesh file.
        """
        # Check the shape of the solution vector
        if solution is not None:
            assert (
                solution.shape[-3] == self.num_patches
                and solution.shape[-2:] == list(self._patches.values())[0].shape
            )
            solution = solution.numpy() if isinstance(solution, tn.Tensor) else solution

        with h5.File(path, "w") as f:
            # Create patch and solution group
            mg = f.create_group("Mesh")

            for patch in self._patches.values():
                # Append patch info to mesh
                mg = patch.save(mg)

            # Save mesh information
            mg.attrs["ID"] = self._id
            mg.attrs["Max Processes"] = self._max_processes
            mg.attrs["Decimals"] = self._decimals
            mg.create_dataset("Bounding Boxes", data=self._bboxes)
            if self._name is not None:
                mg.attrs["Name"] = self._name

            # Boundary connection information
            i = 0
            bcs = mg.create_group("Boundary Connections")
            for key, value in self._boundary_hash.items():
                bc = bcs.create_group(f"Boundary {i}")
                bc.create_dataset("Point", data=key)

                # Patch ids
                pids = [pid for pid in value if pid is not None]
                if pids != []:
                    bc.create_dataset("Patch IDs", data=pids)
                i += 1

            if solution is not None:
                f.create_dataset("Solution", data=solution)

            if k is not None:
                f.attrs["k"] = k

    @classmethod
    def load(cls, path="mesh.hdf5"):
        """
        Load IGA mesh from hdf5 file.

        Parameters
        ----------
        path: str or PathLike
            Location to save the h5 mesh file.

        Returns
        -------
        mesh: ttnte.iga.IGAMesh
            IGA mesh.
        k: float
            The eigenvalue if it was saved.
        solution: numpy.ndarray
            The solution, either angular or scalar flux, if it was saved.
        """
        returns = []
        with h5.File(path, "r") as f:
            mg = f["Mesh"]

            # Create boundary hash
            boundary_hash = {}
            has_reflective_boundary = False
            for bg in mg["Boundary Connections"].values():
                point = tuple(bg["Point"][()])
                boundary_hash[point] = (
                    list(bg["Patch IDs"][()]) if "Patch IDs" in bg else [None, None]
                )

                # Fix vacuum boundaries
                if len(boundary_hash[point]) == 1:
                    boundary_hash[point] = [*boundary_hash[point], None]
                assert len(boundary_hash[point]) == 2

                if boundary_hash[point][0] == boundary_hash[point][1]:
                    has_reflective_boundary = True

            # Create patches
            patches = {}
            for name, pg in mg.items():
                if name.split(" ")[0] == "Patch":
                    patch = Patch.load(pg)
                    patches[patch.id] = patch

            # Create mesh
            mesh = IGAMesh()
            mesh._patches = patches
            mesh._boundary_hash = boundary_hash
            mesh._id = mg.attrs["ID"]
            mesh._name = mg.attrs["Name"] if "Name" in mg.attrs else None
            mesh._max_processes = mg.attrs["Max Processes"]
            mesh._bcs_set = True
            mesh._finalized = True
            mesh._connected = True
            mesh._has_reflective_boundary = has_reflective_boundary
            mesh._decimals = mg.attrs["Decimals"]
            mesh._bboxes = mg["Bounding Boxes"][()]
            returns.append(mesh)

            # Get solution if it exists
            if "Solution" in f:
                returns.append(np.array(f["Solution"][()]))

            if "k" in f.attrs:
                returns.append(f.attrs["k"])

        return returns

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
        for i, patch in enumerate(self._patches.values()):
            patch.set_phi(phi[i,])

    def pid2pidx(self, pid):
        return np.argwhere(pid == np.array(list(self._patches.keys()))).flatten()[0]

    @property
    def num_patches(self):
        return len(self._patches)

    @property
    def patches(self):
        return self._patches

    @property
    def patch_ids(self):
        return np.array(list(self._patches.keys()))

    @property
    def max_processes(self):
        return self._max_processes

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return id

    @property
    def has_reflective_boundary(self):
        return self._has_reflective_boundary

    @max_processes.setter
    def max_processes(self, max_processes):
        self._max_processes = max_processes

    @name.setter
    def name(self, name):
        self._name = name
