import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import factorial
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Union

import cotengra as ctg
import numpy as np
import pandas as pd
import torch as tn
import torchtt as tntt
from scipy.special import lpmv

from ttnte.__init__ import IS_NOTEBOOK
from ttnte.assemblers.operators import (
    FissionOperator,
    ScatteringOperator,
    SparseOperator,
)
from ttnte.cad.patch import Patch
from ttnte.iga import IGAMesh
from ttnte.xs import Server

if IS_NOTEBOOK:
    import tqdm.notebook as tqdm
else:
    import tqdm


@dataclass
class PatchInfo:
    idx: int
    id: int
    patch: Patch
    xknot_intervals: np.ndarray
    yknot_intervals: np.ndarray
    knot_intervals: List[np.ndarray]
    I1: int
    I2: int


@dataclass
class Operators:
    H: Optional[Union[SparseOperator, tntt.TT]] = None
    S: Optional[Union[ScatteringOperator, tntt.TT]] = None
    F: Optional[Union[FissionOperator, tntt.TT]] = None
    q: Optional[tn.Tensor] = None
    B_in: Optional[Union[SparseOperator, tntt.TT]] = None
    B_out: Optional[Union[SparseOperator, tntt.TT]] = None


class MatrixAssembler(object):
    """
    Sparse matrix assembler. All matrices assembled with this will be in
    coordinate sparse format.

    Attributes
    ----------
    shape: list of tuple of int
        The shape of the full size operators.
    N: tuple of int
        N shape of shape.
    M: tuple of int
        M shape of shape.
    avg_element_size: float
        Average size of each knot span.
    """

    _quadrants = np.array(np.meshgrid([1, -1], [1, -1])).T.reshape(-1, 2)

    def __init__(
        self,
        mesh: IGAMesh,
        xs_server: Server,
        num_ordinates: int,
        num_points: Optional[Tuple[int]] = None,
        max_processes: int = max(1, cpu_count() - 1),
    ):
        """
        Initialize MatrixAssembler object.

        Parameters
        ----------
        mesh: ttnte.iga.IGAMesh
            The IGA mesh.
        xs_server: ttnte.xs.Server
            Object for material cross section information.
        num_ordinates: int
            Discretization for discrete ordinates.
        max_processes: int, default=max(1, multiprocessing.cpu_count() - 1)
            Maximum allowed processes.
        """
        self._mesh = mesh
        self._xs_server = xs_server
        self._num_points = num_points
        self._num_ordinates = num_ordinates
        self._max_processes = max_processes
        self._only = ["H", "S", "F", "q", "B_in", "B_out"]

        # Check all patches in IGAMesh have the correct shape and degree
        shape = list(self._mesh.patches.values())[0].shape
        degree = list(self._mesh.patches.values())[0].degree

        for patch in self._mesh.patches.values():
            if shape != patch.shape or degree != patch.degree:
                raise RuntimeError(
                    "All patches in the IGAMesh should have the same "
                    + "shape and degree"
                )

        self._num_points = (
            (degree[0] + 1, degree[1] + 1) if num_points is None else num_points
        )

        # Get angular quadrature
        self._ordinates = self._chebyshev_legendre(self._num_ordinates)

        # Get spatial quadrature
        self._xtilde, self._wx = np.polynomial.legendre.leggauss(self._num_points[0])
        self._ytilde, self._wy = np.polynomial.legendre.leggauss(self._num_points[1])

        # Expected operator shapes
        self._op_shape = 2 * [
            self._quadrants.shape[0],
            self._ordinates[0].shape[0],
            self._ordinates[1].shape[0],
            self._xs_server.num_groups,
            self._mesh.num_patches,
            *shape,
        ]

    # ========================================================================
    # Main build methods

    def _initialize_build(self, verbose):
        self._verbose = verbose
        self._assembly_info = {}
        self._start_time = time.time()

        if self._verbose:
            print(75 * "=")
            print(f"Running {self.__class__.__name__}.build()")
            print(
                "Discretization: N = {}, G = {}, P = {}, A = {}, B = {}".format(
                    self._num_ordinates,
                    self._xs_server.num_groups,
                    self._mesh.num_patches,
                    *next(iter(self._mesh.patches.values())).shape,
                )
            )
            print("Operators: " + ", ".join(self._only))
            print(75 * "-")

    def build(
        self,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Build operators.

        Parameters
        ----------
        verbose: bool, default=True
            Print build progress. This will only be true if ``max_processes=1``.
        **kwargs
            Change which operators to build by specifying ``False`` for any
            of the following operators:

            - ``H``: transport operator,
            - ``S``: scattering operator,
            - ``F``: fission operator,
            - ``q``: fixed source vector,
            - ``B_out``: outward boundary operator,
            - ``B_in``: inward boundary operator.

            .. note::
                Only operators with non-zero entries will be not ``None``.

        Returns
        -------
        operators: ttnte.assemblers.Operators
            Resulting operators.
        """
        ops = self._build(kwargs, verbose)

        # Save and print final data
        self._print_final(ops)

        # Place resulting operators in the correct sparse operators
        return Operators(
            **{
                name: {
                    "H": SparseOperator,
                    "S": lambda S: ScatteringOperator(
                        self._Y, S, self._ordinates[0][:, 0], self._ordinates[1][:, 0]
                    ),
                    "F": lambda F: FissionOperator(
                        F, self._ordinates[0][:, 0], self._ordinates[1][:, 0]
                    ),
                    "q": lambda q: q,
                    "B_in": SparseOperator,
                    "B_out": SparseOperator,
                }[name](op)
                for name, op in ops.items()
            }
        )

    def _build(self, kwargs, verbose):
        get_volumes = any(op in self._only for op in ["H", "S", "F", "q"])
        get_boundaries = "B_in" in self._only or "B_out" in self._only

        # Check which operators to build
        self._check_which_ops(kwargs)

        # Begin build process
        self._initialize_build(verbose)

        # Setup spherical harmonics
        self._setup_sph_harm()

        # Create progress bar
        pbar = tqdm.tqdm(total=self._mesh.num_patches)
        pbar.refresh()

        # Calcualte basis data
        R, dR = self._basis(self._get_patch_info(0)) if get_volumes else (None, None)

        # Get boundary basis data
        bR = self._boundary_basis(self._get_patch_info(0)) if get_boundaries else None

        ops = {}
        with ProcessPoolExecutor(
            max_workers=min(self._max_processes, self._mesh.num_patches * 2)
        ) as executor:
            run_volumes = any(op in self._only for op in ["H", "S", "F", "q"])
            run_boundaries = "B_in" in self._only or "B_out" in self._only
            vlists = []
            blists = []

            # Add tasks to queue
            for i in range(self._mesh.num_patches):
                vlists.append(
                    executor.submit(self._build_patch, i, R, dR) if run_volumes else []
                )
                blists.append(
                    executor.submit(self._build_boundaries, self._get_patch_info(i), bR)
                    if run_boundaries
                    else []
                )

            # Iterate through tasks and add operators
            while len(vlists) > 0:
                vlist = vlists.pop(0).result() if run_volumes else vlists.pop(0)
                blist = blists.pop(0).result() if run_boundaries else blists.pop(0)

                for name, op in zip(self._only, vlist + blist):
                    if op is not None:
                        ops[name] = ops[name] + op if name in ops else op

                del vlist, blist
                pbar.update(1)
                pbar.refresh()

            executor.shutdown(wait=False)

        return ops

    def _check_which_ops(self, kwargs):
        # Check which operators to build
        self._only = []
        for operator, check in zip(
            ["H", "S", "F", "q", "B_out", "B_in"],
            [
                True,
                self._check_scatter(),
                self._check_fission(),
                self._check_source(),
                True,
                self._mesh.num_patches > 1 or self._mesh.has_reflective_boundary,
            ],
        ):
            if kwargs.get(operator, True) and check:
                self._only.append(operator)

    def _check_scatter(self):
        for mat in self._xs_server.materials:
            if self._xs_server.scatter_gtg(mat) is not None:
                return True
        return False

    def _check_fission(self):
        for mat in self._xs_server.materials:
            if self._xs_server.nu_fission(mat) is not None:
                return True
        return False

    def _check_source(self):
        for patch in self._mesh.patches.values():
            if patch.source is not None:
                return True
        return False

    def _get_patch_info(self, i):
        # Get the patch
        patch = list(self._mesh.patches.values())[i]

        # Get knot span intervals
        xknot_intervals = np.unique(patch.knotvectors[0])
        yknot_intervals = np.unique(patch.knotvectors[1])

        return PatchInfo(
            idx=i,
            id=patch.id,
            patch=patch,
            xknot_intervals=xknot_intervals,
            yknot_intervals=yknot_intervals,
            knot_intervals=[xknot_intervals, yknot_intervals],
            I1=xknot_intervals.size - 1,
            I2=yknot_intervals.size - 1,
        )

    def _build_patch(self, i, R, dR=None):
        # Get the patch info
        pinfo = self._get_patch_info(i)

        # Setup current patch
        if self._max_processes == 1:
            self._print_patch(pinfo.id)

        J = None
        if "H" in self._only:
            # Get Jacobian
            J = self._jacobian(pinfo)

        ops = []
        if any(op in self._only for op in ["H", "S", "F", "q"]):
            # Cross-interpolate Jacobian determinant
            J_det = self._jacobian_det(pinfo)

            # Combine J_det and R^T
            JRT = self._JRT(J_det, R)
            del J_det

            # Build local volume integrals
            Intg_int, Intg_str = self._build_local_integrals(
                pinfo, R, JRT, J, dR if "H" in self._only else None
            )
            del J, R, dR

            # Append ops
            if "H" in self._only:
                ops.append(self._build_loss(pinfo, Intg_int, Intg_str))
            del Intg_str

            start = time.time()
            if "S" in self._only:
                ops.append(self._build_scatter(pinfo, Intg_int))

            if "F" in self._only:
                ops.append(self._build_fission(pinfo, Intg_int))

            if "q" in self._only:
                ops.append(
                    self._build_source(
                        pinfo,
                        JRT if isinstance(JRT, tn.Tensor) else JRT.full().squeeze(),
                    )
                )
            del Intg_int

        return ops

    def _build_loss(self, pinfo: PatchInfo, Intg_int, Intg_str):
        """"""
        # Compute dot product with ordinates
        mu = np.ones((2, self._ordinates[0].shape[0]))
        eta = np.ones((2, self._ordinates[1].shape[0]))
        mu[0, :] = self._ordinates[0][:, 1]
        mu[1, :] = np.sqrt(1 - mu[0, :] ** 2)
        eta[1, :] = np.cos(self._ordinates[1][:, 1])

        H = (
            ctg.einsum(
                "qnk,g,abcd->qnkgabcd",
                tn.ones(
                    (
                        self._quadrants.shape[0],
                        self._ordinates[0].shape[0],
                        self._ordinates[1].shape[0],
                    )
                ),
                tn.tensor(self._xs_server.total(pinfo.patch.material)),
                Intg_int,
            ).to_sparse_coo()
            - ctg.einsum(
                "qi,in,ik,g,abicd->qnkgabcd",
                tn.tensor(self._quadrants),
                tn.tensor(mu),
                tn.tensor(eta),
                tn.ones(self._xs_server.num_groups),
                Intg_str,
            ).to_sparse_coo()
        )

        # Add another patch dimension and diagonalize energy and ordinates
        indices = tn.zeros(
            (H.indices().shape[0] + 6, H.indices().shape[1]), dtype=tn.int64
        )
        indices[tn.tensor([0, 1, 2, 3, 5, 6, 12, 13]), :] = H.indices()
        indices[tn.tensor([4, 11]), :] = pinfo.idx
        indices[7:11, :] = indices[:4]

        # Flatten to a matrix and return
        H = self._flatten(
            tn.sparse_coo_tensor(indices, H.values(), size=self._op_shape).coalesce()
        )
        self._append_info("H", H)
        return H

    def _build_scatter(self, pinfo: PatchInfo, Intg_int):
        """"""
        # Check if scattering is zero in this patch
        if self._xs_server.scatter_gtg(pinfo.patch.material) is None:
            return None

        S = ctg.einsum(
            "lij,abcd->liabjcd",
            tn.tensor(self._xs_server.scatter_gtg(pinfo.patch.material)),
            Intg_int,
        ).to_sparse_coo()

        # Add an additional patch dimension
        indices = tn.zeros(
            (S.indices().shape[0] + 2, S.indices().shape[1]), dtype=tn.int64
        )
        indices[tn.tensor([0, 1, 3, 4, 5, 7, 8]), :] = S.indices()
        indices[tn.tensor([2, 6]), :] = pinfo.idx

        # New shape
        shape = (
            *S.shape[:2],
            self._mesh.num_patches,
            *S.shape[2:5],
            self._mesh.num_patches,
            *S.shape[5:],
        )

        S = self._reshape(
            tn.sparse_coo_tensor(indices, S.values(), size=shape).coalesce(),
            (self._xs_server.num_moments, *2 * [np.prod(self._op_shape[-4:])]),
        )
        self._append_info("S", S)
        return S

    def _build_fission(self, pinfo: PatchInfo, Intg_int):
        """"""
        if self._xs_server.nu_fission(pinfo.patch.material) is None:
            return None

        # Compute fission integral
        F = ctg.einsum(
            "i,j,abcd->iabjcd",
            tn.tensor(self._xs_server.chi),
            tn.tensor(self._xs_server.nu_fission(pinfo.patch.material)),
            Intg_int,
        ).to_sparse_coo()

        # Add an additional patch dimension
        indices = tn.zeros(
            (F.indices().shape[0] + 2, F.indices().shape[1]), dtype=tn.int64
        )
        indices[tn.tensor([0, 2, 3, 4, 6, 7]), :] = F.indices()
        indices[tn.tensor([1, 5]), :] = pinfo.idx

        # New shape
        shape = (
            F.shape[0],
            self._mesh.num_patches,
            *F.shape[1:4],
            self._mesh.num_patches,
            *F.shape[-2:],
        )

        F = self._flatten(
            tn.sparse_coo_tensor(indices, F.values(), size=shape).coalesce()
        )
        self._append_info("F", F)
        return F

    def _build_source(self, pinfo: PatchInfo, JRT):
        # Check if the patch has a source
        if pinfo.patch.source is None:
            return None

        # Get sampling positions
        coords = tn.empty((pinfo.I1, pinfo.I2, *self._num_points, 2))

        # Points in each knot span
        j1, j2 = [
            j.flatten()
            for j in np.meshgrid(
                np.arange(self._num_points[0]), np.arange(self._num_points[1])
            )
        ]

        for i1, i2 in itertools.product(range(pinfo.I1), range(pinfo.I2)):
            coords[i1, i2, ...] = self._calc_coords(pinfo, i1, i2, j1, j2).reshape(
                (coords.shape[2:])
            )

        # Evaluate the source at all the coordinates
        source = tn.tensor(
            pinfo.patch.source(coords.reshape((-1, 2)), self._num_ordinates)
        ).reshape(
            (
                self._quadrants.shape[0],
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._xs_server.num_groups,
                *coords.shape[:-1],
            )
        )
        RJq = ctg.einsum("abcdef,qikgabcd->qikgabef", JRT, source)

        # Compute integrated source
        q = tn.zeros(
            (
                self._quadrants.shape[0],
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._xs_server.num_groups,
                self._mesh.num_patches,
                *pinfo.patch.shape,
            )
        )

        for i1, i2 in itertools.product(range(pinfo.I1), range(pinfo.I2)):
            q[
                ...,
                pinfo.idx,
                i1 : i1 + pinfo.patch.degree[0] + 1,
                i2 : i2 + pinfo.patch.degree[1] + 1,
            ] += RJq[..., i1, i2, :, :]

        return q.flatten()

    def _build_local_integrals(self, pinfo: PatchInfo, R, JRT, J=None, dR=None):
        """"""
        # Build components
        RJRT = ctg.einsum("abcdef,abcdgh->abefgh", R, JRT)
        if dR is not None:
            dRJRT = ctg.einsum("abcdef,abcdghf,abcdij->abgheij", J, dR, JRT)

        # Get shape and degree
        pshape = pinfo.patch.shape
        pdegree = pinfo.patch.degree

        # Sum local integrals
        Intg_int = tn.zeros((*pshape, *pshape))
        Intg_str = tn.zeros((*pshape, 2, *pshape)) if dR is not None else None

        for i1, i2 in itertools.product(range(pinfo.I1), range(pinfo.I2)):
            Intg_int[
                i1 : i1 + pdegree[0] + 1,
                i2 : i2 + pdegree[1] + 1,
                i1 : i1 + pdegree[0] + 1,
                i2 : i2 + pdegree[1] + 1,
            ] += RJRT[i1, i2, ...]

            if dR is not None:
                Intg_str[
                    i1 : i1 + pdegree[0] + 1,
                    i2 : i2 + pdegree[1] + 1,
                    :,
                    i1 : i1 + pdegree[0] + 1,
                    i2 : i2 + pdegree[1] + 1,
                ] += dRJRT[i1, i2, ...]

        return Intg_int, Intg_str

    def _JRT(self, J_det, R):
        return ctg.einsum(
            "abcd,c,d,abcdef->abcdef",
            J_det,
            tn.tensor(self._wx),
            tn.tensor(self._wy),
            R,
        )

    # ========================================================================
    # Boundary build methods

    def _build_boundaries(self, pinfo: PatchInfo, R):
        """"""
        # Compute Jacobian of the boundary integral
        J_det = self._boundary_jacobian_det(pinfo)

        ops = []
        if "B_out" in self._only:
            ops.append(self._build_outgoing_boundary(pinfo, J_det, R))

        if "B_in" in self._only:
            ops.append(self._build_incident_boundary(pinfo, J_det, R))

        return ops

    def _build_outgoing_boundary(self, pinfo: PatchInfo, J_det, R):
        """"""
        # Get angular data
        Ox, Oy = self._angular(pinfo, dir=1.0)

        # Get local boundary integrals
        B_out = self._build_boundary_integrals(pinfo, J_det, R, (Ox, Oy), "out")
        self._append_info("B_out", B_out)
        return B_out

    def _build_incident_boundary(self, pinfo: PatchInfo, J_det, R):
        """"""
        # Unpack Jacobians
        Jx_det, Jy_det = J_det

        # Get angular data
        Ox, Oy = self._angular(pinfo, dir=-1.0)

        # Mask vacuum conditions
        for bidx in range(2):
            # Get adjacent patch index
            xp = self._mesh.get_connected_patch(pinfo.id, centroid=(0.5, bidx))
            yp = self._mesh.get_connected_patch(pinfo.id, centroid=(bidx, 0.5))

            if xp is None:
                Jx_det[bidx, ...] = 0
            if yp is None:
                Jy_det[bidx, ...] = 0

        # Get local boundary integrals
        B_in = self._build_boundary_integrals(
            pinfo, (Jx_det, Jy_det), R, (Ox, Oy), "in"
        )
        self._append_info("B_in", B_in)
        return B_in

    def _build_boundary_integrals(self, pinfo: PatchInfo, J_det, R, Or, tag):
        """"""
        Jx_det, Jy_det = J_det
        Rx, Ry = R
        Ox, Oy = Or

        # Calculate local boundary integrals
        local_Intg_x = ctg.einsum(
            "kij,j,qnmkij,kijab,kijcd->qnmkiabcd",
            Jx_det,
            tn.tensor(self._wx),
            Ox,
            Rx,
            Rx,
        )
        local_Intg_y = ctg.einsum(
            "kij,j,qnmkij,kijab,kijcd->qnmkiabcd",
            Jy_det,
            tn.tensor(self._wy),
            Oy,
            Ry,
            Ry,
        )
        del Jx_det, Jy_det, Ox, Oy, Rx, Ry

        return self._concat_boundary_integrals(
            pinfo,
            [0, 1],
            local_Intg_x,
            connected_patches=(
                [self._mesh.get_connected_patch(pinfo.id, (0.5, i)) for i in range(2)]
                if tag == "in"
                else None
            ),
        ) + self._concat_boundary_integrals(
            pinfo,
            [2, 3],
            local_Intg_y,
            connected_patches=(
                [self._mesh.get_connected_patch(pinfo.id, (i, 0.5)) for i in range(2)]
                if tag == "in"
                else None
            ),
        )

    def _concat_boundary_integrals(
        self, pinfo: PatchInfo, bidxs, local_Intg, connected_patches
    ):
        """"""
        Intg = tn.zeros(
            (
                2,
                self._quadrants.shape[0],
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                *pinfo.patch.shape,
                *pinfo.patch.shape,
            )
        )

        # Iterate though subelement adding its contribution
        for bidx, i in itertools.product(
            bidxs, range(pinfo.I1 if bidxs[0] < 2 else pinfo.I2)
        ):
            if bidx < 2:
                i1 = i
                i2 = 0 if bidx == 0 else pinfo.I2 - 1
            else:
                i1 = 0 if bidx == 2 else pinfo.I1 - 1
                i2 = i

            Intg[
                bidx % 2,
                ...,
                i1 : i1 + pinfo.patch.degree[0] + 1,
                i2 : i2 + pinfo.patch.degree[1] + 1,
                i1 : i1 + pinfo.patch.degree[0] + 1,
                i2 : i2 + pinfo.patch.degree[1] + 1,
            ] += local_Intg[:, :, :, bidx % 2, i, ...]

        # Convert to COO
        Intg = Intg.to_sparse_coo()

        # Diagonalize ordinate dimensions and add patch dimension
        indices = tn.zeros(
            (Intg.indices().shape[0] + 5, Intg.indices().shape[1]), dtype=tn.int64
        )
        indices[tn.tensor([0, 1, 2, 3, 5, 6, 11, 12]), :] = Intg.indices()
        indices[tn.tensor([4, 10]), :] = pinfo.idx
        indices[7:10, :] = indices[1:4, :]

        # Handle incident boundary conditions
        if connected_patches is not None:
            for bidx, pid in zip(bidxs, connected_patches):
                if pid == pinfo.id:
                    # Handle reflective boundary condition

                    for quad in range(4):
                        quadrant = self._quadrants[quad, :].copy()

                        # Calculate normal at axis aligned boundary
                        coords = self._calc_boundary_coords(
                            pinfo,
                            bidx,
                            np.array(self._num_points[0 if bidx < 2 else 1] * [0]),
                            np.array(
                                list(range(self._num_points[0 if bidx < 2 else 1]))
                            ),
                        )
                        normals = np.round(
                            pinfo.patch.normal(
                                Patch.centroids[bidx], coords[:, 0 if bidx < 2 else 1]
                            )[-1],
                            6,
                        )

                        # Check reflective boundaries are axis aligned
                        if (normals[:, 0] != normals[0, 0]).all() and (
                            normals[:, 1] != normals[1, 0]
                        ).all():
                            raise RuntimeError(
                                "Reflective boundary conditions must be axis aligned"
                            )

                        # Reflect quadrant ordinates
                        quadrant[1 if normals[0, 0] == 0 else 0] *= -1

                        # Find reflected index
                        mask = (indices[0, :] == bidx % 2) & (indices[1, :] == quad)
                        indices[7, mask] = (
                            (self._quadrants == tuple(quadrant))
                            .all(axis=1)
                            .nonzero()[0][0]
                        )

                else:
                    if pid is not None:
                        # Indicate connected patch
                        mask = (indices[0, :] == bidx % 2) & (
                            indices[4, :] == pinfo.idx
                        )
                        indices[-3, mask] = self._mesh.pid2pidx(pid)

                        # Flip indices
                        indices[-1 if bidx < 2 else -2, mask] = tn.abs(
                            indices[-1 if bidx < 2 else -2, mask]
                            - (
                                pinfo.patch.shape[1] - 1
                                if bidx < 2
                                else pinfo.patch.shape[0] - 1
                            )
                        )

        # Seperate two boundaries
        mask = Intg.indices()[0, :] == bidxs[0]
        Intg0 = tn.sparse_coo_tensor(
            indices[1:, mask],
            Intg.values()[mask],
            size=(
                *2
                * [
                    self._quadrants.shape[0],
                    self._ordinates[0].shape[0],
                    self._ordinates[1].shape[0],
                    self._mesh.num_patches,
                    *pinfo.patch.shape,
                ],
            ),
        ).coalesce()
        Intg1 = tn.sparse_coo_tensor(
            indices[1:, ~mask],
            Intg.values()[~mask],
            size=(
                *2
                * [
                    self._quadrants.shape[0],
                    self._ordinates[0].shape[0],
                    self._ordinates[1].shape[0],
                    self._mesh.num_patches,
                    *pinfo.patch.shape,
                ],
            ),
        ).coalesce()
        Intg = (Intg0 + Intg1).coalesce()
        del Intg0, Intg1

        # Add group index
        indices = tn.zeros(
            (
                Intg.indices().shape[0] + 2,
                self._xs_server.num_groups * Intg.indices().shape[1],
            ),
        )
        indices[tn.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13]), :] = tn.kron(
            tn.ones(self._xs_server.num_groups), Intg.indices()
        )
        indices[3, :] = tn.kron(
            tn.arange(self._xs_server.num_groups), tn.ones(Intg.indices().shape[1])
        )
        indices[10, :] = tn.kron(
            tn.arange(self._xs_server.num_groups), tn.ones(Intg.indices().shape[1])
        )
        Intg = tn.sparse_coo_tensor(
            indices,
            tn.kron(tn.ones(self._xs_server.num_groups), Intg.values()),
            size=self._op_shape,
        ).coalesce()

        # Flatten
        return self._flatten(Intg)

    def _setup_sph_harm(self):
        """"""
        self._Y = []
        for a in range(self._xs_server.num_moments):
            self._Y.append(tn.tensor(self._sph_harm(a)))
        self._Y = tn.concatenate(self._Y, axis=0)

    # ========================================================================
    # Assembly steps

    def _jacobian(self, pinfo: PatchInfo):
        """"""
        # Evaluate Jacobian at each subelement
        J = tn.empty((pinfo.I1, pinfo.I2, *self._num_points, 2, 2))

        for i1, i2 in itertools.product(range(pinfo.I1), range(pinfo.I2)):
            J[i1, i2, ...] = self._calc_jacobian(pinfo, i1, i2)

        return J

    def _jacobian_det(self, pinfo: PatchInfo):
        """"""
        # Evaluate Jacobian determinant at each subelement
        J_det = tn.empty((pinfo.I1, pinfo.I2, *self._num_points))

        for i1, i2 in itertools.product(range(pinfo.I1), range(pinfo.I2)):
            J_det[i1, i2, ...] = self._calc_jacobian_det(pinfo, i1, i2)

        return J_det

    def _basis(self, pinfo):
        """"""
        R = tn.empty(
            (
                pinfo.I1,
                pinfo.I2,
                *self._num_points,
                pinfo.patch.degree[0] + 1,
                pinfo.patch.degree[1] + 1,
            )
        )
        dR = tn.empty(
            (
                pinfo.I1,
                pinfo.I2,
                *self._num_points,
                pinfo.patch.degree[0] + 1,
                pinfo.patch.degree[1] + 1,
                2,
            )
        )

        for i1, i2 in itertools.product(range(pinfo.I1), range(pinfo.I2)):
            R[i1, i2, ...], dR[i1, i2, ...] = self._calc_basis(pinfo, i1, i2)

        return R, dR

    def _boundary_jacobian_det(self, pinfo: PatchInfo):
        """"""
        # Evaluate boundary Jacobian determinant for each subelement
        Jx_det = tn.empty((2, pinfo.I1, self._num_points[0]))
        Jy_det = tn.empty((2, pinfo.I2, self._num_points[1]))

        for bidx, i1 in itertools.product(range(2), range(pinfo.I1)):
            Jx_det[bidx, i1] = self._calc_boundary_jacobian_det(pinfo, bidx, i1)

        for bidx, i2 in itertools.product(range(2), range(pinfo.I2)):
            Jy_det[bidx, i2] = self._calc_boundary_jacobian_det(pinfo, bidx + 2, i2)

        return Jx_det, Jy_det

    def _boundary_basis(self, pinfo: PatchInfo):
        """"""
        # Evaluate boundary Jacobian determinant for each subelement
        Rx = tn.empty(
            (
                2,
                pinfo.I1,
                self._num_points[0],
                pinfo.patch.degree[0] + 1,
                pinfo.patch.degree[1] + 1,
            )
        )
        Ry = tn.empty(
            (
                2,
                pinfo.I2,
                self._num_points[1],
                pinfo.patch.degree[0] + 1,
                pinfo.patch.degree[1] + 1,
            )
        )

        for bidx, i1 in itertools.product(range(2), range(pinfo.I1)):
            Rx[bidx, i1] = self._calc_boundary_basis(pinfo, bidx, i1)

        for bidx, i2 in itertools.product(range(2), range(pinfo.I2)):
            Ry[bidx, i2] = self._calc_boundary_basis(pinfo, bidx + 2, i2)

        return Rx, Ry

    def _angular(self, pinfo, dir):
        """"""
        # Evaluate boundary angular component
        Ox = tn.empty(
            (
                4,
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._num_points[0],
                pinfo.I1,
                2,
            )
        )
        Oy = tn.empty(
            (
                4,
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._num_points[1],
                pinfo.I2,
                2,
            )
        )

        # Calculate and fill
        for bidx, i1 in itertools.product(range(2), range(pinfo.I1)):
            Ox[..., i1, bidx] = self._calc_angular(pinfo, bidx, dir, i1)

        for bidx, i2 in itertools.product(range(2), range(pinfo.I2)):
            Oy[..., i2, bidx] = self._calc_angular(pinfo, bidx + 2, dir, i2)

        # Permute
        Ox = tn.permute(Ox, [0, 1, 2, 5, 4, 3])
        Oy = tn.permute(Oy, [0, 1, 2, 5, 4, 3])

        return Ox, Oy

    def _source(self, pinfo):
        """"""
        # Evaluate internal source
        q = tn.empty(
            (
                pinfo.I1,
                pinfo.I2,
                *self._num_points,
            )
        )

    # ========================================================================
    # Non-interpolation methods

    def _calc_jacobian(self, pinfo: PatchInfo, i1, i2):
        """"""
        # Get indices
        j1, j2 = [
            j.flatten()
            for j in np.meshgrid(
                np.arange(self._num_points[0]), np.arange(self._num_points[1])
            )
        ]
        i1 = (i1 * np.ones(j1.size)).astype(int)
        i2 = (i2 * np.ones(j1.size)).astype(int)

        # Calculate jacobian
        return self._sample_jacobian(pinfo, np.array([i1, i2, j1, j2]).T).reshape(
            (*self._num_points, 2, 2)
        )

    def _calc_jacobian_det(self, pinfo: PatchInfo, i1, i2):
        """"""
        # Get indices
        j1, j2 = [
            j.flatten()
            for j in np.meshgrid(
                np.arange(self._num_points[0]), np.arange(self._num_points[1])
            )
        ]
        i1 = (i1 * np.ones(j1.size)).astype(int)
        i2 = (i2 * np.ones(j1.size)).astype(int)

        # Calculate jacobian
        return self._sample_jacobian_det(pinfo, np.array([i1, i2, j1, j2]).T).reshape(
            self._num_points
        )

    def _calc_basis(self, pinfo: PatchInfo, i1, i2):
        """"""
        # Get indices
        j1, j2 = [
            j.flatten()
            for j in np.meshgrid(
                np.arange(self._num_points[0]), np.arange(self._num_points[1])
            )
        ]
        i1 = (i1 * np.ones(j1.size)).astype(int)
        i2 = (i2 * np.ones(j2.size)).astype(int)

        # Calculate local basis data
        basis_data = self._sample_basis(pinfo, np.array([i1, i2, j1, j2]).T)
        basis_data = basis_data.reshape((*self._num_points, *basis_data.shape[1:-1], 3))

        return basis_data[..., 0], basis_data[..., 1:]

    def _calc_boundary_jacobian_det(self, pinfo: PatchInfo, bidx, i):
        """"""
        # Get indices
        j = np.arange(self._num_points[0 if bidx < 2 else 1])
        i = (i * np.ones(j.size)).astype(int)

        # Calculate Jacobian
        return self._sample_boundary_jacobian_det(pinfo, bidx, np.array([i, j]).T)

    def _calc_boundary_basis(self, pinfo: PatchInfo, bidx, i):
        """"""
        # Get indices
        j = np.arange(self._num_points[0 if bidx < 2 else 1])
        i = (i * np.ones(j.size)).astype(int)

        # Calculate non-vanishing boundary basis functions
        return self._sample_boundary_basis(pinfo, bidx, np.array([i, j]).T)

    def _calc_angular(self, pinfo: PatchInfo, bidx, dir, i):
        """"""
        # Get indices
        j = np.arange(self._num_points[0 if bidx < 2 else 1])
        i = (i * np.ones(j.size)).astype(int)

        # Sample angular component of boundary
        return self._sample_angular(pinfo, bidx, dir, np.array([i, j]).T)

    # ========================================================================
    # Sampling Functions

    def _sample_basis(self, pinfo: PatchInfo, idxs):
        """"""
        # Get indices
        i1, i2, j1, j2 = [idxs[:, i] for i in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_coords(pinfo, i1, i2, j1, j2)

        # Compute local basis for given knots
        basis_data = pinfo.patch.basis_function_grads(coords).reshape(
            (
                coords.shape[0],
                pinfo.patch.degree[0] + 1,
                pinfo.patch.degree[1] + 1,
                3,
            )
        )

        return tn.tensor(basis_data)

    def _sample_boundary_basis(self, pinfo: PatchInfo, bidx, idxs):
        """"""
        i, j = [idxs[:, k] for k in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_boundary_coords(pinfo, bidx, i, j)

        return tn.tensor(pinfo.patch.basis_functions(coords))

    def _sample_jacobian(self, pinfo: PatchInfo, idxs: np.ndarray):
        """"""
        # Get indices
        i1, i2, j1, j2 = [idxs[:, i] for i in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_coords(pinfo, i1, i2, j1, j2)

        J = tn.tensor(pinfo.patch.jacobian(coords))
        J /= (J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]).reshape((-1, 1, 1))
        J[:, 0, 1] *= -1
        J[:, 1, 0] *= -1
        J[:, 0, 0], J[:, 1, 1] = J[:, 1, 1].clone(), J[:, 0, 0].clone()
        return J

    def _sample_jacobian_det(self, pinfo: PatchInfo, idxs):
        """"""
        # Get indices
        i1, i2, j1, j2 = [idxs[:, i] for i in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_coords(pinfo, i1, i2, j1, j2)

        # Calculate pull back from parent to parametric domain
        jacobian = tn.tensor(
            [
                (
                    (pinfo.xknot_intervals[i1[i] + 1] - pinfo.xknot_intervals[i1[i]])
                    * (pinfo.yknot_intervals[i2[i] + 1] - pinfo.yknot_intervals[i2[i]])
                    / 4
                )
                for i in range(i1.shape[0])
            ]
        )

        # Calculate pull back from parametric to physical domain
        Je = tn.tensor(pinfo.patch.jacobian(coords))
        return tn.abs(
            (Je[:, 0, 0] * Je[:, 1, 1] - Je[:, 0, 1] * Je[:, 1, 0]) * jacobian
        )

    def _sample_boundary_jacobian_det(self, pinfo: PatchInfo, bidx, idxs):
        """"""
        # Initialize result
        jacobian = tn.zeros((idxs.shape[0]))

        # Get indices
        i, j = [idxs[:, k] for k in range(idxs.shape[-1])]

        # Get knot interval
        knot_intervals, param_idx, tilde = (
            (pinfo.xknot_intervals, 0, self._xtilde)
            if bidx < 2
            else (pinfo.yknot_intervals, 1, self._ytilde)
        )

        coords = (
            np.zeros((idxs.shape[0], 2))
            if (bidx % 2) == 0
            else np.ones((idxs.shape[0], 2))
        )
        for k in range(idxs.shape[0]):
            # Calculate coordinates
            coords[k, param_idx] = self.parent2parametric(
                tilde[j[k]], knot_intervals[i[k]], knot_intervals[i[k] + 1]
            )

            # Calculate pull back from parent to parametric domain
            jacobian[k] = (knot_intervals[i[k] + 1] - knot_intervals[i[k]]) / 2

        # Calculate pull back from parametric to physical domain
        Je = tn.tensor(pinfo.patch.jacobian(coords))[:, param_idx, :] ** 2
        Je = tn.sqrt(tn.sum(Je, -1))
        return tn.abs(Je * jacobian)

    def _sample_angular(self, pinfo: PatchInfo, bidx, dir, idxs):
        """"""
        # Get in
        i, j = [idxs[:, k] for k in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_boundary_coords(pinfo, bidx, i, j)

        # Calculate normals at coordinates
        _, normals = pinfo.patch.normal(
            Patch.centroids[bidx], coords[:, 0 if bidx < 2 else 1]
        )

        # Compute dot product with ordinates0.711% 2
        mu = np.ones((2, self._ordinates[0].shape[0]))
        eta = np.ones((2, self._ordinates[1].shape[0]))
        mu[0, :] = self._ordinates[0][:, 1]
        mu[1, :] = np.sqrt(1 - self._ordinates[0][:, 1] ** 2)
        eta[1, :] = np.cos(self._ordinates[1][:, 1])

        products = ctg.einsum(
            "qj,jm,jn,ij->qmni",
            self._quadrants,
            mu,
            eta,
            normals,
        )
        products[(dir * products) < 0] = 0
        return tn.abs(tn.tensor(products))

    # ========================================================================
    # Mappings

    def _calc_coords(self, pinfo, i1, i2, j1, j2):
        """"""
        # Calculate coordinates in parametric dimension
        coords = np.zeros((j1.shape[0], 2))
        coords[:, 0] = self.parent2parametric(
            self._xtilde[j1],
            pinfo.xknot_intervals[i1],
            pinfo.xknot_intervals[i1 + 1],
        )
        coords[:, 1] = self.parent2parametric(
            self._ytilde[j2],
            pinfo.yknot_intervals[i2],
            pinfo.yknot_intervals[i2 + 1],
        )

        return tn.tensor(coords)

    def _calc_boundary_coords(self, pinfo: PatchInfo, bidx, i_idxs, j_idxs):
        """"""
        # Get knot interval
        knot_intervals, param_idx, tilde = (
            (pinfo.xknot_intervals, 0, self._xtilde)
            if bidx < 2
            else (pinfo.yknot_intervals, 1, self._ytilde)
        )

        # Map quadrature to local parametric coordinates
        coords = (
            np.zeros((len(tilde), 2)) if (bidx % 2) == 0 else np.ones((len(tilde), 2))
        )
        coords[j_idxs, param_idx] = self.parent2parametric(
            tilde[j_idxs], knot_intervals[i_idxs], knot_intervals[i_idxs + 1]
        )

        return tn.tensor(coords)

    @staticmethod
    def parent2parametric(tilde, hat_left, hat_right):
        """"""
        return hat_left + (tilde + 1) * (hat_right - hat_left) / 2

    # ========================================================================
    # Sparse matrix operations

    @staticmethod
    def _flatten(A: tn.Tensor):
        """"""
        assert A.is_sparse

        # Flatten into matrix
        new_shape = tuple(2 * [np.prod(A.shape[: int(A.ndim / 2)])])

        return MatrixAssembler._reshape(A, new_shape)

    @staticmethod
    def _reshape(A: tn.Tensor, shape: tuple):
        """"""
        assert A.is_sparse and np.prod(A.shape) == np.prod(shape)

        # Flatten indices
        inds = np.ravel_multi_index(A.indices().numpy(), A.shape)
        inds = np.unravel_index(inds, shape)

        return tn.sparse_coo_tensor(
            tn.tensor(np.array(inds), dtype=tn.int64), A.values(), size=shape
        ).coalesce()

    @staticmethod
    def compression(A: tn.Tensor):
        """"""
        assert A.is_sparse
        entries = np.prod(A.indices().shape) + A.values().shape[0]
        if entries != 0:
            return np.prod(A.shape) / entries
        else:
            return np.inf

    def _append_info(self, name, A, final=False):
        """"""
        # Calculate the number of non-zero entries
        entries = (
            (np.prod(A.indices().shape) + A.values().shape[0])
            if A.is_sparse
            else np.prod(A.shape)
        )

        # Get info
        info = {
            "shape": A.shape,
            "entries": entries,
            "compression": self.compression(A) if A.is_sparse else 1,
            "elapsed time": time.time() - self._start_time,
        }
        if final:
            self._assembly_info[name] = info

        # Print info
        if self._verbose and (final or self._max_processes == 1):
            self._print_info(name, info, final)

    # ========================================================================
    # I/O

    def _print_patch(self, pid):
        if self._verbose:
            print(f"Assembling Patch {pid}")
            print(
                "{:15s} {:25s} {:10s}  {:15s}".format(
                    "Step", "Shape", "Compression", "Elapsed Time (s)"
                )
            )

    def _print_info(self, name, data, final):
        if not final:
            print(
                "{:15s} {:25s} {:10.2f}  {:10.2f}".format(
                    name,
                    ",".join(map(str, data["shape"])),
                    data["compression"],
                    data["elapsed time"],
                )
            )
        else:
            print(
                "{:15s} {:25s} {:10.2f}".format(
                    name,
                    ",".join(map(str, data["shape"])),
                    data["compression"],
                )
            )

    def _print_final(self, ops):
        if self._verbose:
            print(75 * "-")
            print(
                "Final Operators (Elapsed Time: {} s)".format(
                    round(time.time() - self._start_time, 2)
                )
            )
            print("{:15s} {:25s} {:10s}".format("Step", "Ranks", "Compression"))
        for name, op in ops.items():
            self._append_info(name, op, final=True)
        if self._verbose:
            print(75 * "=")

    def save_info(self, path, round_data=False):
        """
        Save operator info such as name, shape, ranks, number of entries, compression,
        and elapsed time.

        Parameters
        ----------
        path: str
            Path to save CSV file.
        round_data: bool, default=False
            Round compression and elapsed time to 2 and 3 decimal places,
            respectively.
        """
        info = pd.DataFrame(self._assembly_info).transpose()

        # Round numbers
        if round_data:
            info["compression"] = info["compression"].apply(lambda x: round(x, 2))
            info["elapsed time"] = info["elapsed time"].apply(lambda x: round(x, 3))

        info.index.name = "Name"
        info.columns = [s.capitalize() for s in info.columns[:-1]] + [
            "Elapsed Time (s)"
        ]
        info.to_csv(path)

    # ========================================================================
    # Quadrature sets

    @staticmethod
    def _chebyshev_legendre(N):
        """
        Chebyshev-Legendre (square) quadrature set given by
        https://www.osti.gov/servlets/purl/5958402.
        """
        assert N % 4 == 0

        # Number of ordinates for each dimension
        n = np.round(np.sqrt(N)).astype(int)

        # Compute quadrature
        q_l = MatrixAssembler._gauss_legendre(n)
        q_c = MatrixAssembler._gauss_chebyshev(n)

        # Check correct number of ordinates
        assert 4 * q_l.shape[0] * q_c.shape[0] == N

        # Ensure correct weighted sum
        q_l[:, 0] /= 4 * np.outer(q_l[:, 0], q_c[:, 0]).sum()

        return q_l, q_c

    @staticmethod
    def _gauss_legendre(N):
        """
        Gauss Legendre quadrature set.

        Only given for positive half space.
        """
        mu, w = np.polynomial.legendre.leggauss(N)
        w = w[: int(mu.size / 2)] / 2
        mu = np.abs(mu[: int(mu.size / 2)])

        assert np.round(np.sum(w), 1) == 0.5

        return np.concatenate([w[:, np.newaxis], mu[:, np.newaxis]], axis=1)

    @staticmethod
    def _gauss_chebyshev(N):
        """Gauss-Chebyshev quadrature."""
        gamma = (2 * np.arange(1, int(N / 2) + 1) - 1) * np.pi / (2 * N)
        w = np.ones(int(N / 2)) / (2 * N)

        return np.concatenate([w[:, np.newaxis], gamma[:, np.newaxis]], axis=1)

    # ========================================================================
    # Angular moment methods

    def _sph_harm(self, n):
        """"""
        # Ordinates
        mu = ctg.einsum("q,n->qn", self._quadrants[:, 0], self._ordinates[0][:, 1])
        gamma = np.arccos(
            ctg.einsum(
                "q,n->qn", self._quadrants[:, 1], np.cos(self._ordinates[1][:, 1])
            )
        )

        # Calculate even and odd spherical harmonics
        Yl = []
        for m in np.arange(n + 1):
            Ym = (
                (-1) ** m
                * np.sqrt((2 * n + 1) * factorial(n - abs(m)) / factorial(n + abs(m)))
                * ctg.einsum(
                    "qn,qm->qnm",
                    lpmv(m, n, mu),
                    np.cos(m * gamma),
                )
            )

            Yl.append(Ym)

        Yl = np.array(Yl)
        Yl[1:,] *= 2
        return Yl

    # ========================================================================
    # Others

    def angular_integral(self, psi: tn.Tensor):
        """
        Integrate angular dependence in angular flux to get scalar flux.

        Parameters
        ----------
        psi: torch.Tensor
            Angular flux control variables.

        Returns
        -------
        phi: torch.Tensor
            Scalar flux control variables.
        """
        shape = (
            self._xs_server.num_groups,
            self._mesh.num_patches,
            *list(self._mesh.patches.values())[0].shape,
        )
        return ctg.einsum(
            "abcd,b,c->d",
            psi.reshape(
                (
                    self._quadrants.shape[0],
                    self._ordinates[0].shape[0],
                    self._ordinates[1].shape[0],
                    -1,
                )
            ),
            tn.tensor(self._ordinates[0][:, 0]),
            tn.tensor(self._ordinates[1][:, 0]),
        ).reshape(shape)

    def outward_current(self, psi: tn.Tensor):
        """
        Calculate the outward current for the system.

        Parameters
        ----------
        psi: torch.Tensor
            Angular flux control variables.

        Returns
        -------
        Jout: float
            Total outward current.
        """
        # Get solution in the correct shape
        psi = psi.reshape(
            (
                self._quadrants.shape[0],
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._xs_server.num_groups,
                self._mesh.num_patches,
                *list(self._mesh.patches.values())[0].shape,
            )
        )

        # Get basis functions
        Rx, Ry = self._boundary_basis(self._get_patch_info(0))

        # Outgoing current
        Jout = 0

        # Iterate through patches
        for i in range(self._mesh.num_patches):
            # Get patch info
            pinfo = self._get_patch_info(i)

            # Get mask
            mask = tn.zeros(4)

            # Check if the patch has a vacuum boundary
            for j, centroid in enumerate(Patch.centroids):
                if self._mesh.get_connected_patch(
                    pinfo.id, centroid
                ) is None and self._mesh.check_boundary_existance(pinfo.id, centroid):
                    mask[j] = 1

            if mask.sum() == 0:
                continue

            # Get Jacobian information
            Jx_det, Jy_det = self._boundary_jacobian_det(pinfo)

            # Get angular data
            Ox, Oy = self._angular(pinfo, dir=1.0)

            # Calculate local boundary integrals
            local_Intg_x = ctg.einsum(
                "kij,j,qnmkij,kijab->qnmkiab",
                Jx_det,
                tn.tensor(self._wx),
                Ox,
                Rx,
            )
            local_Intg_y = ctg.einsum(
                "kij,j,qnmkij,kijab->qnmkiab",
                Jy_det,
                tn.tensor(self._wy),
                Oy,
                Ry,
            )
            del Jx_det, Jy_det, Ox, Oy

            # Iterate through x boundaries
            psi_bx = tn.concatenate(
                [
                    psi[
                        ...,
                        i,
                        :,
                        : pinfo.patch.degree[1] + 1,
                    ].unsqueeze_(-1),
                    psi[
                        ...,
                        i,
                        :,
                        pinfo.I2 - 1 : pinfo.I2 + pinfo.patch.degree[1],
                    ].unsqueeze_(-1),
                ],
                axis=-1,
            )
            for i1 in range(pinfo.I1):
                Jout += ctg.einsum(
                    "k,qnmkab,n,m,qnmgabk->",
                    mask[:2],
                    local_Intg_x[:, :, :, :, i1, :, :],
                    tn.tensor(self._ordinates[0][:, 0]),
                    tn.tensor(self._ordinates[1][:, 0]),
                    psi_bx[..., i1 : i1 + pinfo.patch.degree[0] + 1, :, :],
                )

            # Iterate through y boundaries
            psi_by = tn.concatenate(
                [
                    psi[..., i, : pinfo.patch.degree[0] + 1, :].unsqueeze_(-1),
                    psi[
                        ...,
                        i,
                        pinfo.I1 - 1 : pinfo.I1 + pinfo.patch.degree[0],
                        :,
                    ].unsqueeze_(-1),
                ],
                axis=-1,
            )
            for i2 in range(pinfo.I2):
                Jout += ctg.einsum(
                    "k,qnmkab,n,m,qnmgabk->",
                    mask[2:],
                    local_Intg_y[:, :, :, :, i2, :, :],
                    tn.tensor(self._ordinates[0][:, 0]),
                    tn.tensor(self._ordinates[1][:, 0]),
                    psi_by[..., :, i2 : i2 + pinfo.patch.degree[1] + 1, :],
                )

        return Jout

    def total_production(self):
        """
        Get total production for a fixed source problem.

        Returns
        -------
        P: float
            Total production from internal source.
        """
        # Initialize production
        P = 0

        for i in range(self._mesh.num_patches):
            # Get patch info
            pinfo = self._get_patch_info(i)

            # Check if there's an internal source
            if pinfo.patch.source is None:
                continue

            # Get sampling positions
            coords = tn.empty((pinfo.I1, pinfo.I2, *self._num_points, 2))

            # Points in each knot span
            j1, j2 = [
                j.flatten()
                for j in np.meshgrid(
                    np.arange(self._num_points[0]), np.arange(self._num_points[1])
                )
            ]

            for i1, i2 in itertools.product(range(pinfo.I1), range(pinfo.I2)):
                coords[i1, i2, ...] = self._calc_coords(pinfo, i1, i2, j1, j2).reshape(
                    (coords.shape[2:])
                )

            # Get the Jacobian
            J_det = self._jacobian_det(pinfo)

            # Sample the source
            source = tn.tensor(
                pinfo.patch.source(coords.reshape((-1, 2)), self._num_ordinates)
            ).reshape(
                (
                    self._quadrants.shape[0],
                    self._ordinates[0].shape[0],
                    self._ordinates[1].shape[0],
                    self._xs_server.num_groups,
                    *coords.shape[:-1],
                )
            )

            P += ctg.einsum(
                "abcd,c,d,n,m,qnmgabcd->",
                J_det,
                tn.tensor(self._wx),
                tn.tensor(self._wy),
                tn.tensor(self._ordinates[0][:, 0]),
                tn.tensor(self._ordinates[1][:, 0]),
                source,
            )

        return P

    # ========================================================================
    # Getters

    @property
    def N(self) -> Tuple[int]:
        N = np.array(
            [
                self._quadrants.shape[0],
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._xs_server.num_groups,
                self._mesh.num_patches,
                *list(self._mesh.patches.values())[0].shape,
            ]
        ).astype(int)
        return tuple(N[N > 1])

    @property
    def M(self):
        return self.N

    @property
    def shape(self):
        return [(self.N[i], self.N[i]) for i in range(len(self.N))]

    # @property
    # def avg_element_size(self):
    #     # Turn off printing
    #     verbose = self._verbose
    #     self._verbose = False
    #
    #     # Iterate through patches
    #     size = 0
    #     for p in range(self._mesh.num_patches):
    #         # Set the current patch
    #         self._setup_current_patch(p)
    #
    #         # Cross-interpolate Jacobian determinant
    #         J_det = self._jacobian_det()
    #
    #         # Calculate basis data at quadrature points for each knot span
    #         R, _ = self._basis()
    #
    #         # Add contribution
    #         size += 1 / (self._I1 * self._I2) * ctg.einsum("abcdef,abcd->", R, J_det)
    #
    #     # Reset verbose
    #     self._verbose = verbose
    #
    #     return size / self._mesh.num_patches
