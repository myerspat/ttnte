import itertools
import time
from math import factorial
from typing import Optional, Tuple

import cotengra as ctg
import numpy as np
import torch as tn
from scipy.special import lpmv

from ttnte.iga import IGAMesh
from ttnte.xs import Server

from .operators import FissionOperator, ScatteringOperator, SparseOperator


class MatrixAssembler(object):
    def __init__(
        self,
        mesh: IGAMesh,
        xs_server: Server,
        num_ordinates: int,
        num_points: Optional[Tuple[int]] = None,
    ):
        """"""
        self._mesh = mesh
        self._xs_server = xs_server
        self._num_ordinates = num_ordinates
        self._num_points = (
            (mesh.degree[0] + 1, mesh.degree[1] + 1)
            if num_points is None
            else num_points
        )
        self._quadrants = np.array(np.meshgrid([1, -1], [1, -1])).T.reshape(-1, 2)

    # ========================================================================
    # Main build methods

    def _initialize_build(self, verbose):
        """"""
        self._verbose = verbose
        self._assembly_info = {}
        self._start_time = time.time()

        if self._verbose:
            print(
                "Discretization: N = {}, G = {}, P = {}, A = {}, B = {}".format(
                    self._num_ordinates,
                    self._xs_server.num_groups,
                    self._mesh.num_patches,
                    self._mesh.patches[0].ctrlpts_size_u,
                    self._mesh.patches[0].ctrlpts_size_v,
                )
            )

        # Get angular quadrature
        self._ordinates = self._chebyshev_legendre(self._num_ordinates)

    def _setup_current_patch(self, p):
        """"""
        # Print patch
        self._print_patch(p)

        # Get current patch
        self._p = p
        self._patch = self._mesh.patches[p]

        # Get knot span intervals
        self._xknot_intervals = np.unique(self._patch.knotvector_u)
        self._yknot_intervals = np.unique(self._patch.knotvector_v)
        self._knot_intervals = [self._xknot_intervals, self._yknot_intervals]

        # Get number of knot spans along each parametric dimension
        self._I1 = self._xknot_intervals.size - 1
        self._I2 = self._yknot_intervals.size - 1

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
            self._patch.ctrlpts_size_u,
            self._patch.ctrlpts_size_v,
        ]

    def build(self, verbose: bool = True):
        """"""
        # Begin build process
        self._initialize_build(verbose)

        # Setup spherical harmonics
        self._setup_sph_harm()

        # Concatenate operators
        H, S, F, B_in, B_out = self._build_patch(0)

        for p in range(1, self._mesh.num_patches):
            ops = self._build_patch(p)
            H += ops[0]
            S += ops[1]
            F += ops[2]
            B_in += ops[3]
            B_out += ops[4]

        # Check matrices are sparse
        assert H.is_sparse and S.is_sparse and F.is_sparse and B_out.is_sparse

        # Save sparse matrix info
        if self._verbose:
            print("\nFinal Operators")
        self._append_coo_info("H", H)
        self._append_coo_info("S", S)
        self._append_coo_info("F", F)
        self._append_coo_info("B_in", B_out)
        self._append_coo_info("B_out", B_out)

        return (
            SparseOperator(H),
            ScatteringOperator(
                self._Y, S, self._ordinates[0][:, 0], self._ordinates[1][:, 0]
            ),
            FissionOperator(F, self._ordinates[0][:, 0], self._ordinates[1][:, 0]),
            SparseOperator(B_in),
            SparseOperator(B_out),
        )

    def _build_patch(self, p):
        # Setup current patch
        self._setup_current_patch(p)

        # Get Jacobian
        J = self._jacobian()

        # Cross-interpolate Jacobian determinant
        J_det = self._jacobian_det()

        # Calculate basis data at quadrature points for each knot span
        R, dR = self._basis()

        # Build local volume integrals
        Intg = self._build_local_integrals(J, J_det, R, dR)
        del J, J_det, R, dR

        # Buil operators
        H = self._build_loss(Intg[:, :, 0, :, :], Intg[:, :, 1:, :, :])
        self._append_coo_info("H", H)
        S = self._build_scatter(Intg[:, :, 0, :, :])
        self._append_coo_info("S", S)
        F = self._build_fission(Intg[:, :, 0, :, :])
        self._append_coo_info("F", F)
        B_in = self._build_outgoing_boundary()
        self._append_coo_info("B_in", B_in)
        B_out = self._build_outgoing_boundary()
        self._append_coo_info("B_out", B_out)

        return H, S, F, B_in, B_out

    def _build_loss(self, Intg_int, Intg_str):
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
                tn.tensor(self._xs_server.total(self._patch.name)),
                Intg_int,
            ).to_sparse_coo()
            + ctg.einsum(
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
        indices[tn.tensor([4, 11]), :] = self._p
        indices[7:11, :] = indices[:4]

        # Flatten to a matrix and return
        H = self._flatten(
            tn.sparse_coo_tensor(indices, H.values(), size=self._op_shape).coalesce()
        )
        return H

    def _build_scatter(self, Intg_int):
        """"""
        S = ctg.einsum(
            "lij,abcd->liabjcd",
            tn.tensor(self._xs_server.scatter_gtg(self._patch.name)),
            Intg_int,
        ).to_sparse_coo()

        # Add an additional patch dimension
        indices = tn.zeros(
            (S.indices().shape[0] + 2, S.indices().shape[1]), dtype=tn.int64
        )
        indices[tn.tensor([0, 1, 3, 4, 5, 7, 8]), :] = S.indices()
        indices[tn.tensor([2, 6]), :] = self._p

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

        return S

    def _build_fission(self, Intg_int):
        """"""
        # Compute fission integral
        F = ctg.einsum(
            "i,j,abcd->iabjcd",
            tn.tensor(self._xs_server.chi),
            tn.tensor(self._xs_server.nu_fission(self._patch.name)),
            Intg_int,
        ).to_sparse_coo()

        # Add an additional patch dimension
        indices = tn.zeros(
            (F.indices().shape[0] + 2, F.indices().shape[1]), dtype=tn.int64
        )
        indices[tn.tensor([0, 2, 3, 4, 6, 7]), :] = F.indices()
        indices[tn.tensor([1, 5]), :] = self._p

        # New shape
        shape = (
            F.shape[0],
            self._mesh.num_patches,
            *F.shape[1:4],
            self._mesh.num_patches,
            *F.shape[-2:],
        )

        return self._flatten(
            tn.sparse_coo_tensor(indices, F.values(), size=shape).coalesce()
        )

    def _build_local_integrals(self, J, J_det, R, dR):
        """"""
        # Build components
        JRT = ctg.einsum(
            "abcd,c,d,abcdef->abcdef",
            J_det,
            tn.tensor(self._wx),
            tn.tensor(self._wy),
            R,
        )
        RJRT = ctg.einsum("abcdef,abcdgh->abefgh", R, JRT)
        dRJRT = ctg.einsum("abcdef,abcdghf,abcdij->abgheij", J, dR, JRT)
        del JRT

        # Sum local integrals
        Intg = tn.zeros(
            (
                self._patch.ctrlpts_size_u,
                self._patch.ctrlpts_size_v,
                3,
                self._patch.ctrlpts_size_u,
                self._patch.ctrlpts_size_v,
            )
        )

        for i1, i2 in itertools.product(range(self._I1), range(self._I2)):
            Intg[
                i1 : i1 + self._patch.degree_u + 1,
                i2 : i2 + self._patch.degree_v + 1,
                0,
                i1 : i1 + self._patch.degree_u + 1,
                i2 : i2 + self._patch.degree_v + 1,
            ] += RJRT[i1, i2, ...]
            Intg[
                i1 : i1 + self._patch.degree_u + 1,
                i2 : i2 + self._patch.degree_v + 1,
                1:,
                i1 : i1 + self._patch.degree_u + 1,
                i2 : i2 + self._patch.degree_v + 1,
            ] += dRJRT[i1, i2, ...]

        return Intg

    def _build_outgoing_boundary(self):
        """"""
        # Compute Jacobian of the boundary integral
        Jx_det, Jy_det = self._boundary_jacobian_det()

        # Get basis data
        Rx, Ry = self._boundary_basis()

        # Get angular data
        Ox, Oy = self._angular(dir=1.0)

        # Get local boundary integrals
        return self._build_boundary_integrals(
            (Jx_det, Jy_det), (Rx, Ry), (Ox, Oy), "out"
        )

    def _build_incident_boundary(self):
        """"""
        # Compute Jacobian of the boundary integral
        Jx_det, Jy_det = self._boundary_jacobian_det()

        # Get basis data
        Rx, Ry = self._boundary_basis()

        # Get angular data
        Ox, Oy = self._angular(dir=-1.0)

        # Mask vacuum conditions
        for boundary_idx in range(2):
            # Get adjacent patch index
            xp = self._mesh.get_connected_patch(self._p, coord=(0.5, boundary_idx))
            yp = self._mesh.get_connected_patch(self._p, coord=(boundary_idx, 0.5))

            if xp is None:
                Jx_det[boundary_idx, ...] = 0
            if yp is None:
                Jy_det[boundary_idx, ...] = 0

        # Get local boundary integrals
        return self._build_boundary_integrals(
            (Jx_det, Jy_det), (Rx, Ry), (Ox, Oy), "in"
        )

    def _build_boundary_integrals(self, J_det, R, Or, tag):
        """"""
        Jx_det, Jy_det = J_det
        Rx, Ry = R
        Ox, Oy = Or

        # Calculate local boundary integrals
        local_Intg_x = tn.einsum(
            "kij,j,qnmkij,kijab,kijcd->qnmkiabcd",
            (Jx_det, tn.tensor(self._wx), Ox, Rx, Rx),
        )
        local_Intg_y = tn.einsum(
            "kij,j,qnmkij,kijab,kijcd->qnmkiabcd",
            (Jy_det, tn.tensor(self._wy), Oy, Ry, Ry),
        )
        del Jx_det, Jy_det, Ox, Oy, Rx, Ry

        return self._concat_boundary_integrals(
            [0, 1],
            local_Intg_x,
            connected_patches=(
                [self._mesh.get_connected_patch(self._p, (0.5, i)) for i in range(2)]
                if tag == "in"
                else None
            ),
        ) + self._concat_boundary_integrals(
            [2, 3],
            local_Intg_y,
            connected_patches=(
                [self._mesh.get_connected_patch(self._p, (i, 0.5)) for i in range(2)]
                if tag == "in"
                else None
            ),
        )

    def _concat_boundary_integrals(self, boundary_idxs, local_Intg, connected_patches):
        """"""
        Intg = tn.zeros(
            (
                2,
                self._quadrants.shape[0],
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._patch.ctrlpts_size_u,
                self._patch.ctrlpts_size_v,
                self._patch.ctrlpts_size_u,
                self._patch.ctrlpts_size_v,
            )
        )

        # Iterate though subelement adding its contribution
        for boundary_idx, i in itertools.product(
            boundary_idxs, range(self._I1 if boundary_idxs[0] < 2 else self._I2)
        ):
            if boundary_idx < 2:
                i1 = i
                i2 = 0 if boundary_idx == 0 else self._I2 - 1
            else:
                i1 = 0 if boundary_idx == 2 else self._I1 - 1
                i2 = i

            Intg[
                boundary_idx % 2,
                ...,
                i1 : i1 + self._patch.degree_u + 1,
                i2 : i2 + self._patch.degree_v + 1,
                i1 : i1 + self._patch.degree_u + 1,
                i2 : i2 + self._patch.degree_v + 1,
            ] = local_Intg[:, :, :, boundary_idx % 2, i, ...]

        # Convert to COO
        Intg = Intg.to_sparse_coo()

        # Diagonalize ordinate dimensions and add patch dimension
        indices = tn.zeros(
            (Intg.indices().shape[0] + 5, Intg.indices().shape[1]), dtype=tn.int64
        )
        indices[tn.tensor([0, 1, 2, 3, 5, 6, 11, 12]), :] = Intg.indices()
        indices[tn.tensor([4, 10]), :] = self._p
        indices[7:10, :] = indices[1:4, :]

        # Handle incident boundary conditions
        if connected_patches is not None:
            for boundary_idx, p in zip(boundary_idxs, connected_patches):
                if p == self._p:
                    # Handle reflective boundary condition

                    for quad in range(4):
                        quadrant = self._quadrants[quad, :].copy()

                        # Calculate normal at axis aligned boundary
                        coords = self._calc_boundary_coords(
                            boundary_idx,
                            self._num_points[0 if boundary_idx < 2 else 1] * [0],
                            list(range(self._num_points[0 if boundary_idx < 2 else 1])),
                        )
                        normals = np.round(self._mesh.normal(self._p, coords)[-1], 6)

                        # Check reflective boundaries are axis aligned
                        if (normals[0, :] != normals[0, 0]).all() and (
                            normals[1, :] != normals[1, 0]
                        ).all():
                            raise RuntimeError(
                                "Reflective boundary conditions must be axis aligned"
                            )

                        # Reflect quadrant ordinates
                        quadrant[1 if normals[0, 0] == 0 else 0] *= -1

                        # Find reflected index
                        indices[
                            :,
                            (indices[0, :] == boundary_idx % 2)
                            & (indices[1, :] == quad),
                        ][7, :] = (
                            (self._quadrants == tuple(quadrant))
                            .all(axis=1)
                            .nonzero()[0][0]
                        )

                else:
                    if p is not None:
                        # Indicate connected patch
                        indices[
                            :,
                            (indices[0, :] == boundary_idx % 2)
                            & (indices[4, :] == self._p),
                        ][-3, :] = p

                        # Flip indices
                        indices[
                            :,
                            (indices[0, :] == boundary_idx % 2)
                            & (indices[4, :] == self._p),
                        ][-1 if boundary_idx < 2 else -2, :] = tn.abs(
                            indices[
                                :,
                                (indices[0, :] == boundary_idx % 2)
                                & (indices[4, :] == self._p),
                            ][-1 if boundary_idx < 2 else -2, :]
                            - (
                                self._patch.ctrlpts_size_v
                                if boundary_idx < 2
                                else self._patch.ctrlpts_size_u - 1
                            )
                        )

        # Seperate two boundaries
        Intg0 = tn.sparse_coo_tensor(
            indices[1:, (indices[0, :] == boundary_idxs[0])],
            Intg.values()[(indices[0, :] == boundary_idxs[0])],
            size=(
                *2
                * [
                    self._quadrants.shape[0],
                    self._ordinates[0].shape[0],
                    self._ordinates[1].shape[0],
                    self._mesh.num_patches,
                    self._patch.ctrlpts_size_u,
                    self._patch.ctrlpts_size_v,
                ],
            ),
        ).coalesce()
        Intg1 = tn.sparse_coo_tensor(
            indices[1:, (indices[0, :] == boundary_idxs[1])],
            Intg.values()[(indices[0, :] == boundary_idxs[1])],
            size=(
                *2
                * [
                    self._quadrants.shape[0],
                    self._ordinates[0].shape[0],
                    self._ordinates[1].shape[0],
                    self._mesh.num_patches,
                    self._patch.ctrlpts_size_u,
                    self._patch.ctrlpts_size_v,
                ],
            ),
        ).coalesce()
        Intg = Intg0 + Intg1
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

    def _jacobian(self):
        """"""
        # Evaluate Jacobian at each subelement
        J = tn.empty((self._I1, self._I2, *self._num_points, 2, 2))

        for i1, i2 in itertools.product(range(self._I1), range(self._I2)):
            J[i1, i2, ...] = self._calc_jacobian(i1, i2)

        return J

    def _jacobian_det(self):
        """"""
        # Evaluate Jacobian determinant at each subelement
        J_det = tn.empty((self._I1, self._I2, *self._num_points))

        for i1, i2 in itertools.product(range(self._I1), range(self._I2)):
            J_det[i1, i2, ...] = self._calc_jacobian_det(i1, i2)

        return J_det

    def _basis(self):
        """"""
        R = tn.empty(
            (
                self._I1,
                self._I2,
                *self._num_points,
                self._patch.degree_u + 1,
                self._patch.degree_v + 1,
            )
        )
        dR = tn.empty(
            (
                self._I1,
                self._I2,
                *self._num_points,
                self._patch.degree_u + 1,
                self._patch.degree_v + 1,
                2,
            )
        )

        for i1, i2 in itertools.product(range(self._I1), range(self._I2)):
            R[i1, i2, ...], dR[i1, i2, ...] = self._calc_basis(i1, i2)

        return R, dR

    def _boundary_jacobian_det(self):
        """"""
        # Evaluate boundary Jacobian determinant for each subelement
        Jx_det = tn.empty((2, self._I1, self._num_points[0]))
        Jy_det = tn.empty((2, self._I2, self._num_points[1]))

        for boundary_idx, i1 in itertools.product(range(2), range(self._I1)):
            Jx_det[boundary_idx, i1] = self._calc_boundary_jacobian_det(
                boundary_idx, i1
            )
        for boundary_idx, i2 in itertools.product(range(2), range(self._I2)):
            Jy_det[boundary_idx, i2] = self._calc_boundary_jacobian_det(
                boundary_idx + 2, i2
            )

        return Jx_det, Jy_det

    def _boundary_basis(self):
        """"""
        # Evaluate boundary Jacobian determinant for each subelement
        Rx = tn.empty(
            (
                2,
                self._I1,
                self._num_points[0],
                self._patch.degree_u + 1,
                self._patch.degree_v + 1,
            )
        )
        Ry = tn.empty(
            (
                2,
                self._I2,
                self._num_points[1],
                self._patch.degree_u + 1,
                self._patch.degree_v + 1,
            )
        )

        for boundary_idx, i1 in itertools.product(range(2), range(self._I1)):
            Rx[boundary_idx, i1] = self._calc_boundary_basis(boundary_idx, i1)
        for boundary_idx, i2 in itertools.product(range(2), range(self._I2)):
            Ry[boundary_idx, i2] = self._calc_boundary_basis(boundary_idx + 2, i2)

        return Rx, Ry

    def _angular(self, dir):
        """"""
        # Evaluate boundary angular component
        Ox = tn.empty(
            (
                4,
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._num_points[0],
                self._I1,
                2,
            )
        )
        Oy = tn.empty(
            (
                4,
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._num_points[1],
                self._I2,
                2,
            )
        )

        # Calculate and fill
        for boundary_idx, i1 in itertools.product(range(2), range(self._I1)):
            Ox[..., i1, boundary_idx] = self._calc_angular(boundary_idx, dir, i1)
        for boundary_idx, i2 in itertools.product(range(2), range(self._I2)):
            Oy[..., i2, boundary_idx] = self._calc_angular(boundary_idx + 2, dir, i2)

        # Permute
        Ox = tn.permute(Ox, [0, 1, 2, 5, 4, 3])
        Oy = tn.permute(Oy, [0, 1, 2, 5, 4, 3])

        return Ox, Oy

    # ========================================================================
    # Non-interpolation methods

    def _calc_jacobian(self, i1, i2):
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
        return tn.permute(
            self._sample_jacobian(np.array([i1, i2, j1, j2]).T), (2, 0, 1)
        ).reshape((*self._num_points, 2, 2))

    def _calc_jacobian_det(self, i1, i2):
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
        return self._sample_jacobian_det(np.array([i1, i2, j1, j2]).T).reshape(
            self._num_points
        )

    def _calc_basis(self, i1, i2):
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
        basis_data = self._sample_basis(np.array([i1, i2, j1, j2]).T)
        basis_data = basis_data.reshape((*self._num_points, *basis_data.shape[1:-1], 3))

        return basis_data[..., 0], basis_data[..., 1:]

    def _calc_boundary_jacobian_det(self, boundary_idx, i):
        """"""
        # Get indices
        j = np.arange(self._num_points[0 if boundary_idx < 2 else 1])
        i = (i * np.ones(j.size)).astype(int)

        # Calculate Jacobian
        return self._sample_boundary_jacobian_det(boundary_idx, np.array([i, j]).T)

    def _calc_boundary_basis(self, boundary_idx, i):
        """"""
        # Get indices
        j = np.arange(self._num_points[0 if boundary_idx < 2 else 1])
        i = (i * np.ones(j.size)).astype(int)

        # Calculate non-vanishing boundary basis functions
        return self._sample_boundary_basis(boundary_idx, np.array([i, j]).T)

    def _calc_angular(self, boundary_idx, dir, i):
        """"""
        # Get indices
        j = np.arange(self._num_points[0 if boundary_idx < 2 else 1])
        i = (i * np.ones(j.size)).astype(int)

        # Sample angular component of boundary
        return self._sample_angular(boundary_idx, dir, np.array([i, j]).T)

    # ========================================================================
    # Sampling Functions

    def _sample_basis(self, idxs):
        """"""
        # Get indices
        i1, i2, j1, j2 = [idxs[:, i] for i in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_coords(i1, i2, j1, j2)

        # Compute local basis for given knots
        basis_data = self._mesh.basis_function_grads(self._p, coords).reshape(
            (
                *coords.shape[1:],
                self._patch.degree_u + 1,
                self._patch.degree_v + 1,
                3,
            )
        )

        return tn.tensor(basis_data)

    def _sample_boundary_basis(self, boundary_idx, idxs):
        """"""
        i, j = [idxs[:, k] for k in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_boundary_coords(boundary_idx, i, j)

        return tn.tensor(self._mesh.basis_functions(self._p, coords))

    def _sample_jacobian(self, idxs):
        """"""
        # Get indices
        i1, i2, j1, j2 = [idxs[:, i] for i in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_coords(i1, i2, j1, j2)

        J = tn.tensor(self._mesh.jacobian(self._p, coords))
        J /= J[0, 0, :] * J[1, 1, :] - J[0, 1, :] * J[1, 0, :]
        J[0, 1, :] *= -1
        J[1, 0, :] *= -1
        J[0, 0, :], J[1, 1, :] = J[1, 1, :].clone(), J[0, 0, :].clone()
        return J

    def _sample_jacobian_det(self, idxs):
        """"""
        # Get indices
        i1, i2, j1, j2 = [idxs[:, i] for i in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_coords(i1, i2, j1, j2)

        # Calculate pull back from parent to parametric domain
        jacobian = tn.tensor(
            [
                (
                    (self._xknot_intervals[i1[i] + 1] - self._xknot_intervals[i1[i]])
                    * (self._yknot_intervals[i2[i] + 1] - self._yknot_intervals[i2[i]])
                    / 4
                )
                for i in range(i1.shape[0])
            ]
        )

        # Calculate pull back from parametric to physical domain
        Je = tn.tensor(self._mesh.jacobian(self._p, coords))
        return tn.abs(
            (Je[0, 0, :] * Je[1, 1, :] - Je[0, 1, :] * Je[1, 0, :]) * jacobian
        )

    def _sample_boundary_jacobian_det(self, boundary_idx, idxs):
        """"""
        # Initialize result
        jacobian = tn.zeros((idxs.shape[0]))

        # Get indices
        i, j = [idxs[:, k] for k in range(idxs.shape[-1])]

        # Get knot interval
        knot_intervals, param_idx, tilde = (
            (self._xknot_intervals, 0, self._xtilde)
            if boundary_idx < 2
            else (self._yknot_intervals, 1, self._ytilde)
        )

        coords = (
            np.zeros((2, idxs.shape[0]))
            if (boundary_idx % 2) == 0
            else np.ones((2, idxs.shape[0]))
        )
        for k in range(idxs.shape[0]):
            # Calculate coordinates
            coords[param_idx, k] = self.parent2parametric(
                tilde[j[k]], knot_intervals[i[k]], knot_intervals[i[k] + 1]
            )

            # Calculate pull back from parent to parametric domain
            jacobian[k] = (knot_intervals[i[k] + 1] - knot_intervals[i[k]]) / 2

        # Calculate pull back from parametric to physical domain
        Je = self._mesh.jacobian(self._p, coords)
        Je = tn.tensor(self._mesh.jacobian(self._p, coords))[param_idx, :, :] ** 2
        Je = tn.sqrt(tn.sum(Je, 0))
        return tn.abs(Je * jacobian)

    def _sample_angular(self, boundary_idx, dir, idxs):
        """"""
        # Get in
        i, j = [idxs[:, k] for k in range(idxs.shape[-1])]

        # Map parent to parametric
        coords = self._calc_boundary_coords(boundary_idx, i, j)

        # Calculate normals at coordinates
        _, normals = self._mesh.normal(self._p, coords)

        # Compute dot product with ordinates
        mu = np.ones((2, self._ordinates[0].shape[0]))
        eta = np.ones((2, self._ordinates[1].shape[0]))
        mu[0, :] = self._ordinates[0][:, 1]
        mu[1, :] = np.sqrt(1 - self._ordinates[0][:, 1] ** 2)
        eta[1, :] = np.cos(self._ordinates[1][:, 1])

        products = ctg.einsum(
            "qj,jm,jn,ji->qmni",
            self._quadrants,
            mu,
            eta,
            normals,
        )
        products[(dir * products) < 0] = 0
        return tn.abs(tn.tensor(products))

    # ========================================================================
    # Mappings

    def _calc_coords(self, i1, i2, j1, j2):
        """"""
        # Calculate coordinates in parametric dimension
        coords = np.zeros((2, i1.shape[0]))
        for i in range(i1.shape[0]):
            coords[0, i] = self.parent2parametric(
                self._xtilde[j1[i]],
                self._xknot_intervals[i1[i]],
                self._xknot_intervals[i1[i] + 1],
            )
            coords[1, i] = self.parent2parametric(
                self._ytilde[j2[i]],
                self._yknot_intervals[i2[i]],
                self._yknot_intervals[i2[i] + 1],
            )

        return tn.tensor(coords)

    def _calc_boundary_coords(self, boundary_idx, i_idxs, j_idxs):
        """"""
        # Get knot interval
        knot_intervals, param_idx, tilde = (
            (self._xknot_intervals, 0, self._xtilde)
            if boundary_idx < 2
            else (self._yknot_intervals, 1, self._ytilde)
        )

        # Map quadrature to local parametric coordinates
        coords = (
            np.zeros((2, len(tilde)))
            if (boundary_idx % 2) == 0
            else np.ones((2, len(tilde)))
        )
        for i, j in zip(i_idxs, j_idxs):
            coords[param_idx, j] = self.parent2parametric(
                tilde[j], knot_intervals[i], knot_intervals[i + 1]
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

    def _append_coo_info(self, name, A):
        """"""
        entries = np.prod(A.indices().shape) + A.values().shape[0]
        self._assembly_info[name] = {
            "shape": A.shape,
            "entries": entries,
            "compression": self.compression(A),
            "elapsed time": time.time() - self._start_time,
        }

        if self._verbose:
            self._print_coo(name)

    # ========================================================================
    # I/O

    def _print_patch(self, p):
        """"""
        if self._verbose:
            print(f"Assembling Patch {p + 1}")
            print(
                "{:15s} {:25s} {:10s}  {:15s}".format(
                    "Step", "Shape", "Compression", "Elapsed Time (s)"
                )
            )

    def _print_coo(self, name):
        """"""
        print(
            "{:15s} {:25s} {:10.2f}  {:10.2f}".format(
                name,
                ",".join(map(str, self._assembly_info[name]["shape"])),
                self._assembly_info[name]["compression"],
                self._assembly_info[name]["elapsed time"],
            )
        )

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
        ).reshape(psi.shape[3:])

    # ========================================================================
    # Getters

    @property
    def N(self):
        N = np.array(
            [
                self._quadrants.shape[0],
                self._ordinates[0].shape[0],
                self._ordinates[1].shape[0],
                self._xs_server.num_groups,
                self._mesh.num_patches,
                self._mesh.patches[0].ctrlpts_size_u,
                self._mesh.patches[0].ctrlpts_size_v,
            ]
        )
        return tuple(N[N > 1])

    @property
    def M(self):
        return self.N

    @property
    def shape(self):
        return [(self.N[i], self.N[i]) for i in range(len(self.N))]
