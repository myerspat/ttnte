import itertools
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch as tn
import torchtt as tntt
from mesh import IGAMesh

from tt_nte.methods.discrete_ordinates import DiscreteOrdinates
from tt_nte.xs import Server


class TT(tntt.TT):
    """torchtt.TT object with general matvec method."""

    def matvec(self, x):
        x = x.reshape(self.__N)
        return self.__matmul__(x).reshape((-1, 1))


class OperatorBuilder(object):
    def __init__(
        self,
        mesh: IGAMesh,
        xs_server: Server,
        num_ordinates: int,
        num_points: Optional[Tuple[int]] = None,
        fixed_source: bool = False,
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
        self._fixed_source = fixed_source
        self._quadrants = np.array(np.meshgrid([1, -1], [1, -1])).T.reshape(-1, 2)

        # Assembly settings
        self.interp_jacobian = True
        self.interp_jacobian_det = True
        self.interp_basis = False

        self.interp_boundary_jacobian_det = True
        self.interp_boundary_basis = False
        self.interp_angular = False

        # Set default precision
        tn.set_default_dtype(tn.float64)

    # ========================================================================
    # Main build methods

    def _setup_current_patch(self, p):
        """"""
        if self._verbose:
            print(f"Assembling Patch {p}")
            print(
                "{:15s} {:25s} {:10s}  {:15s}".format(
                    "Step", "Ranks", "Compression", "Elapsed Time (s)"
                )
            )

        # Get current patch
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

    def build(self, eps: float = 1e-10, verbose: bool = True, use_tt: bool = True):
        """"""
        self._eps = eps
        self._use_tt = use_tt
        self._verbose = verbose
        self._assembly_info = {}
        self._p = 0
        self._start_time = time.time()

        if self._verbose:
            print(
                "Discretization: N = {}, G = {}, A = {}, B = {}".format(
                    self._num_ordinates,
                    self._xs_server.num_groups,
                    self._mesh.patches[0].ctrlpts_size_u,
                    self._mesh.patches[0].ctrlpts_size_v,
                )
            )

        # Get angular quadrature
        self._ordinates = DiscreteOrdinates._compute_square_set(self._num_ordinates, 2)
        self._ordinates[:, 0] /= 4 * np.sum(self._ordinates[:, 0])

        # Setup current patch
        self._setup_current_patch(self._p)

        # Get Jacobian
        J = self._jacobian()

        # Cross-interpolate Jacobian determinant
        J_det = self._jacobian_det()

        # Calculate basis data at quadrature points for each knot span
        R, dR = self._basis()

        if self._use_tt:
            # Save TT info
            self._append_tt_info("J", J)
            self._append_tt_info("J_det", J_det)
            self._append_tt_info("R", R)
            self._append_tt_info("dR", dR)

            # Append basis function support matrices
            J_det = tntt.TT(
                J_det.cores
                + [
                    tn.ones(self._patch.degree_u + 1).reshape((1, -1, 1)),
                    tn.ones(self._patch.degree_v + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )

            # Hadamard product with the weights
            J_det.cores[2] *= tn.tensor(self._wx).reshape((1, -1, 1))
            J_det.cores[3] *= tn.tensor(self._wy).reshape((1, -1, 1))
            J_det = J_det.round(self._eps)

            # Calculate (J_det * R^T)
            J_detRT = tntt.fast_hadammard(J_det, R, self._eps).to_ttm()
            J_detRT.set_core(4, tn.permute(J_detRT.cores[4], (0, 2, 1, 3)))
            J_detRT.set_core(5, tn.permute(J_detRT.cores[5], (0, 2, 1, 3)))
            J_detRT = J_detRT.round(self._eps)
            del J_det
            self._append_tt_info("J_detRT", J_detRT)

            # Calculate J @ dR
            JdR = (
                self.diagonalize(J, np.arange(len(J.cores) - 1))
                .fast_matvec(dR, self._eps)
                .round(self._eps)
            )
            del J, dR
            self._append_tt_info("JdR", JdR)
            JdR = self.diagonalize(JdR, [0, 1, len(JdR.cores) - 1])

            # Transpose quadrature dimensions for sum
            R = self.diagonalize(R, [0, 1])
            R.set_core(2, tn.permute(R.cores[2], (0, 2, 1, 3)))
            R.set_core(3, tn.permute(R.cores[3], (0, 2, 1, 3)))
            JdR.set_core(2, tn.permute(JdR.cores[2], (0, 2, 1, 3)))
            JdR.set_core(3, tn.permute(JdR.cores[3], (0, 2, 1, 3)))

            # Calculate local volume integrals
            Intg_int = self._calc_Intg_x(
                tntt.TT(
                    tntt.amen_mm(R, J_detRT, nswp=50, eps=self._eps)
                    .round(self._eps)
                    .cores
                    + [tn.ones(1).reshape((1, 1, 1, 1))]
                )
            )
            Intg_str = self._calc_Intg_x(
                tntt.amen_mm(
                    JdR,
                    tntt.TT(J_detRT.cores + [tn.ones(2).reshape((1, 2, 1, 1))]),
                    nswp=50,
                    eps=self._eps,
                ).round(self._eps)
            )
            del R, JdR, J_detRT

            # Apply squeeze to remove 1x1 cores
            Intg_int.reduce_dims()
            Intg_str.reduce_dims()

        else:
            JRT = tn.einsum(
                "abcd,c,d,abcdef->abcdef",
                (J_det, tn.tensor(self._wx), tn.tensor(self._wy), R),
            )
            RJRT = tn.einsum("abcdef,abcdgh->abefgh", (R, JRT))
            dRJRT = tn.einsum("abcdef,abcdghf,abcdij->abgheij", (J, dR, JRT))
            del J_det, R, dR, JRT

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

            Intg_int, Intg_str = tntt.TT(
                Intg[:, :, 0, :, :],
                shape=[(Intg.shape[i], Intg.shape[i]) for i in range(2)],
                eps=self._eps,
            ), tntt.TT(
                Intg[:, :, 1:, :, :],
                shape=[(Intg.shape[i], Intg.shape[i]) for i in range(2)] + [[2, 1]],
                eps=self._eps,
            )
            del Intg

        # Build operators with remaining cores
        self._append_tt_info("Intg_int", Intg_int)
        self._append_tt_info("Intg_str", Intg_str)
        H, S, F = self._build_LHS(Intg_int, Intg_str), *self._build_RHS(Intg_int)

        self._append_tt_info("H", H)
        self._append_tt_info("S", S)
        self._append_tt_info("F", F)

        return H, S, F

    def _build_outgoing_boundary(self):
        """"""
        # Compute Jacobian of the boundary integral
        Jx_det, Jy_det = self._boundary_jacobian_det()

        # Get basis data
        Rx, Ry = self._boundary_basis()

        # Get angular data
        Ox, Oy = self._angular(dir=1.0)

        if self._use_tt:
            # Save TT information
            self._append_tt_info("Jx_det_out", Jx_det)
            self._append_tt_info("Jy_det_out", Jy_det)
            self._append_tt_info("Rx_out", Rx)
            self._append_tt_info("Ry_out", Ry)
            self._append_tt_info("Ox_out", Oy)
            self._append_tt_info("Oy_out", Oy)

        # Build boundary integral
        local_Intg_x, local_Intg_y = self._build_boundary_Intg(
            (Jx_det, Jy_det), (Rx, Ry), (Ox, Oy), "out"
        )

        if self._use_tt:
            Intg = (
                self._calc_Intg_bound([0, 1], local_Intg_x)
                + self._calc_Intg_bound([2, 3], local_Intg_y)
            ).round(self._eps)
            Intg.reduce_dims()

        else:
            Intg = tn.zeros(
                (
                    4,
                    self._ordinates.shape[0],
                    self._patch.ctrlpts_size_u,
                    self._patch.ctrlpts_size_v,
                    self._patch.ctrlpts_size_u,
                    self._patch.ctrlpts_size_v,
                )
            )

            # Fill array based on subelement index
            for boundary_idx, i1 in itertools.product(range(2), range(self._I1)):
                i2 = 0 if boundary_idx == 0 else self._I2 - 1
                Intg[
                    ...,
                    i1 : i1 + self._patch.degree_u + 1,
                    i2 : i2 + self._patch.degree_v + 1,
                    i1 : i1 + self._patch.degree_u + 1,
                    i2 : i2 + self._patch.degree_v + 1,
                ] += local_Intg_x[:, :, boundary_idx, i1, ...]
            for boundary_idx, i2 in itertools.product(range(2), range(self._I2)):
                i1 = 0 if boundary_idx == 0 else self._I1 - 1
                Intg[
                    ...,
                    i1 : i1 + self._patch.degree_u + 1,
                    i2 : i2 + self._patch.degree_v + 1,
                    i1 : i1 + self._patch.degree_u + 1,
                    i2 : i2 + self._patch.degree_v + 1,
                ] += local_Intg_y[:, :, boundary_idx, i2, ...]

            # Create TTs
            Intg = tntt.TT(
                Intg.reshape((*Intg.shape[:-2], 1, 1, *Intg.shape[-2:])),
                shape=[(Intg.shape[i], 1) for i in range(2)]
                + [(Intg.shape[i], Intg.shape[i]) for i in range(2, 4)],
                eps=self._eps,
            )

        # Save boundary integral TT info
        self._append_tt_info("Intg_bound_out", Intg)

        # Append energy identity matrix and permute TT
        H_bound = tntt.permute(
            tntt.TT(
                [tn.eye(self._xs_server.num_groups).unsqueeze_(0).unsqueeze_(-1)]
                + Intg.cores,
                eps=0,
            ),
            [1, 2, 0, 3, 4],
            eps=self._eps,
        ).round(self._eps)
        H_bound.reduce_dims()
        self._append_tt_info("H_bound_out", H_bound)
        return self.diagonalize(H_bound, [0, 1])

    def _build_incident_boundary(self):
        """"""
        # Compute Jacobian of the boundary integral
        Jx_det, Jy_det = self._boundary_jacobian_det()

        # Get basis data
        Rx, Ry = self._boundary_basis()

        # Get angular data
        Ox, Oy = self._angular(dir=-1.0)

        # Apply mask to vacuum boundary conditions
        if self._use_tt:
            for boundary_idx in range(2):
                xboundary = self._mesh.get_boundary(self._p, 1, boundary_idx)
                yboundary = self._mesh.get_boundary(self._p, 0, boundary_idx)

                # Make mask
                mask = tn.tensor([0, 1] if boundary_idx == 0 else [1, 0]).reshape(
                    (1, -1, 1)
                )

                if xboundary.from_patch is None:
                    # Apply mask
                    Jx_det.set_core(0, Jx_det.cores[0].clone() * mask)
                    Rx.set_core(0, Rx.cores[0].clone() * mask)
                    Ox.set_core(2, Ox.cores[2].clone() * mask)

                if yboundary.from_patch is None:
                    # Apply mask
                    Jy_det.set_core(0, Jy_det.cores[0].clone() * mask)
                    Ry.set_core(0, Ry.cores[0].clone() * mask)
                    Oy.set_core(2, Oy.cores[2].clone() * mask)

            # Round results
            Jx_det, Jy_det = Jx_det.round(self._eps), Jy_det.round(self._eps)
            Rx, Ry = Rx.round(self._eps), Ry.round(self._eps)
            Ox, Oy = Ox.round(self._eps), Oy.round(self._eps)

            # Save TT information
            self._append_tt_info("Jx_det_in", Jx_det)
            self._append_tt_info("Jy_det_in", Jy_det)
            self._append_tt_info("Rx_in", Rx)
            self._append_tt_info("Ry_in", Ry)
            self._append_tt_info("Ox_in", Oy)
            self._append_tt_info("Oy_in", Oy)

        else:
            for boundary_idx in range(2):
                xboundary = self._mesh.get_boundary(self._p, 1, boundary_idx)
                yboundary = self._mesh.get_boundary(self._p, 0, boundary_idx)

                if xboundary.from_patch is None:
                    Jx_det[boundary_idx, ...] = 0
                if yboundary.from_patch is None:
                    Jy_det[boundary_idx, ...] = 0

        # Build boundary operator
        local_Intg_x, local_Intg_y = self._build_boundary_Intg(
            (Jx_det, Jy_det), (Rx, Ry), (Ox, Oy), "in"
        )

        if self._use_tt:
            Intg = (
                self._calc_Intg_bound(
                    [0, 1],
                    local_Intg_x,
                    incident_boundaries=[
                        self._mesh.get_boundary(self._p, 1, i) for i in range(2)
                    ],
                )
                + self._calc_Intg_bound(
                    [2, 3],
                    local_Intg_y,
                    incident_boundaries=[
                        self._mesh.get_boundary(self._p, 0, i) for i in range(2)
                    ],
                )
            ).round(self._eps)
            Intg.reduce_dims()

        else:
            raise NotImplementedError()

        # Save boundary integral TT info
        self._append_tt_info("Intg_bound_in", Intg)

        H_bound = tntt.permute(
            tntt.TT(
                [tn.eye(self._xs_server.num_groups).unsqueeze_(0).unsqueeze_(-1)]
                + Intg.cores,
                eps=0,
            ),
            [1, 2, 0, 3, 4],
            eps=self._eps,
        ).round(self._eps)
        H_bound.reduce_dims()
        self._append_tt_info("H_bound_in", H_bound)
        return self.diagonalize(H_bound, [1])

    def _build_boundary_Intg(self, J_det, R, On, tag):
        """"""
        Jx_det, Jy_det = J_det
        Rx, Ry = R
        Ox, Oy = On

        if self._use_tt:
            # Add angular component to basis
            Rx = tntt.TT(
                [
                    tn.ones(self._quadrants.shape[0]).reshape((1, -1, 1)),
                    tn.ones(self._ordinates.shape[0]).reshape((1, -1, 1)),
                ]
                + Rx.cores,
                eps=0,
            )
            Ry = tntt.TT(
                [
                    tn.ones(self._quadrants.shape[0]).reshape((1, -1, 1)),
                    tn.ones(self._ordinates.shape[0]).reshape((1, -1, 1)),
                ]
                + Ry.cores,
                eps=0,
            )

            # Add basis component to angular
            Ox = tntt.TT(
                Ox.cores
                + [
                    tn.ones(self._patch.degree_u + 1).reshape((1, -1, 1)),
                    tn.ones(self._patch.degree_v + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )
            Oy = tntt.TT(
                Oy.cores
                + [
                    tn.ones(self._patch.degree_u + 1).reshape((1, -1, 1)),
                    tn.ones(self._patch.degree_v + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )

            # Hadamard product with angular component
            ORx = tntt.fast_hadammard(Ox, Rx, self._eps)
            ORy = tntt.fast_hadammard(Oy, Ry, self._eps)
            del Ox, Oy
            self._append_tt_info(f"ORx_{tag}", ORx)
            self._append_tt_info(f"ORy_{tag}", ORy)

            # Hadamard product with the weights
            Jx_det.cores[-1] *= tn.tensor(self._wx).reshape((1, -1, 1))
            Jy_det.cores[-1] *= tn.tensor(self._wy).reshape((1, -1, 1))

            # Add angular and basis cores to Jacobian
            Jx_det = tntt.TT(
                [
                    tn.ones(self._quadrants.shape[0]).reshape((1, -1, 1)),
                    tn.ones(self._ordinates.shape[0]).reshape((1, -1, 1)),
                ]
                + Jx_det.cores
                + [
                    tn.ones(self._patch.degree_u + 1).reshape((1, -1, 1)),
                    tn.ones(self._patch.degree_v + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )
            Jy_det = tntt.TT(
                [
                    tn.ones(self._quadrants.shape[0]).reshape((1, -1, 1)),
                    tn.ones(self._ordinates.shape[0]).reshape((1, -1, 1)),
                ]
                + Jy_det.cores
                + [
                    tn.ones(self._patch.degree_u + 1).reshape((1, -1, 1)),
                    tn.ones(self._patch.degree_v + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )

            # Hadamard product with Jacobian
            ORJx_det = tntt.fast_hadammard(ORx, Jx_det, self._eps)
            ORJy_det = tntt.fast_hadammard(ORy, Jy_det, self._eps)
            del Jx_det, Jy_det
            self._append_tt_info(f"ORJx_det_{tag}", ORJx_det)
            self._append_tt_info(f"ORJy_det_{tag}", ORJy_det)

            # Diagonalize angular and local region components
            ORJx_det = self.diagonalize(ORJx_det, np.arange(len(ORJx_det.cores) - 3))
            ORJy_det = self.diagonalize(ORJy_det, np.arange(len(ORJy_det.cores) - 3))

            # Transpose quadrature for local sum
            ORJx_det.set_core(4, tn.permute(ORJx_det.cores[4], (0, 2, 1, 3)))
            ORJy_det.set_core(4, tn.permute(ORJy_det.cores[4], (0, 2, 1, 3)))

            # Transpose basis functions
            ORTx = ORx.to_ttm()
            ORTx.set_core(5, tn.permute(ORTx.cores[5], (0, 2, 1, 3)))
            ORTx.set_core(6, tn.permute(ORTx.cores[6], (0, 2, 1, 3)))
            ORTy = ORy.to_ttm()
            ORTy.set_core(5, tn.permute(ORTy.cores[5], (0, 2, 1, 3)))
            ORTy.set_core(6, tn.permute(ORTy.cores[6], (0, 2, 1, 3)))
            del ORx, ORy

            # Calculate local surface integral
            local_Intg_x = tntt.amen_mm(ORJx_det, ORTx, nswp=50, eps=self._eps).round(
                self._eps
            )
            local_Intg_y = tntt.amen_mm(ORJy_det, ORTy, nswp=50, eps=self._eps).round(
                self._eps
            )

        else:
            # Calculate local spatial integrals
            local_Intg_x = tn.einsum(
                "kij,j,qnkij,kijab,kijcd->qnkiabcd",
                (Jx_det, tn.tensor(self._wx), Ox, Rx, Rx),
            )
            local_Intg_y = tn.einsum(
                "kij,j,qnkij,kijab,kijcd->qnkiabcd",
                (Jy_det, tn.tensor(self._wy), Oy, Ry, Ry),
            )
            del Jx_det, Ox, Rx
            del Jy_det, Oy, Ry

        return local_Intg_x, local_Intg_y

    def _build_LHS(self, Intg_x, streaming_Intg_x):
        """"""
        # Add angular and energy dimensions
        H = (
            tntt.TT(
                [
                    tn.eye(self._quadrants.shape[0]).unsqueeze_(0).unsqueeze_(-1),
                    tn.eye(self._ordinates.shape[0]).unsqueeze_(0).unsqueeze_(-1),
                    tn.diag(tn.tensor(self._xs_server.total(self._patch.name)))
                    .unsqueeze_(0)
                    .unsqueeze_(-1),
                ]
                + Intg_x.cores
                + [tn.ones(1).reshape((1, 1, 1, 1))],
                eps=0,
            )
            - tntt.TT(
                [
                    tn.diag(tn.tensor(self._quadrants[:, 0]))
                    .unsqueeze_(0)
                    .unsqueeze_(-1),
                    tn.diag(tn.tensor(self._ordinates[:, 1]))
                    .unsqueeze_(0)
                    .unsqueeze_(-1),
                    tn.eye(self._xs_server.num_groups).unsqueeze_(0).unsqueeze_(-1),
                ]
                + streaming_Intg_x.cores[:-1]
                + [streaming_Intg_x.cores[-1][:, [0], :, :]],
                eps=0,
            )
            - tntt.TT(
                [
                    tn.diag(tn.tensor(self._quadrants[:, 1]))
                    .unsqueeze_(0)
                    .unsqueeze_(-1),
                    tn.diag(tn.tensor(self._ordinates[:, 2]))
                    .unsqueeze_(0)
                    .unsqueeze_(-1),
                    tn.eye(self._xs_server.num_groups).unsqueeze_(0).unsqueeze_(-1),
                ]
                + streaming_Intg_x.cores[:-1]
                + [streaming_Intg_x.cores[-1][:, [1], :, :]],
                eps=0,
            )
        )

        # Remove single size dimensions
        H.reduce_dims()

        # Get boundary term
        boundary = self._build_outgoing_boundary()

        # Add outgoing boundary operator
        return (H + boundary).round(self._eps)

    def _build_RHS(self, Intg_x):
        """"""
        # Scattering operator
        S = tntt.TT(
            [
                tn.ones(2 * [self._quadrants.shape[0]]).unsqueeze_(0).unsqueeze_(-1),
                tn.outer(
                    tn.ones(self._ordinates.shape[0]),
                    tn.tensor(self._ordinates[:, 0]),
                )
                .unsqueeze_(0)
                .unsqueeze_(-1),
                tn.tensor(self._xs_server.scatter_gtg(self._patch.name)[0,])
                .unsqueeze_(0)
                .unsqueeze_(-1),
            ]
            + Intg_x.cores,
            eps=0,
        ).round(self._eps)
        S.reduce_dims()

        # Calculate incident bounary operator
        boundary = self._build_incident_boundary()

        # Fission operator
        F = tntt.TT(
            [
                tn.ones(2 * [self._quadrants.shape[0]]).unsqueeze_(0).unsqueeze_(-1),
                tn.outer(
                    tn.ones(self._ordinates.shape[0]),
                    tn.tensor(self._ordinates[:, 0]),
                )
                .unsqueeze_(0)
                .unsqueeze_(-1),
                tn.outer(
                    tn.tensor(self._xs_server.chi),
                    tn.tensor(self._xs_server.nu_fission(self._patch.name)),
                )
                .unsqueeze_(0)
                .unsqueeze_(-1),
            ]
            + Intg_x.cores,
            eps=0,
        ).round(self._eps)
        F.reduce_dims()

        # return S, F
        return (S + boundary).round(self._eps), F

    # ========================================================================
    # Assembly steps

    def _jacobian(self):
        """"""
        if self._use_tt:
            if self.interp_jacobian:
                # Interpolate Jacobian matrix at each quadrature point
                def _interp_jacobian(i, j):
                    indicator = tn.zeros((1, 2, 2, 1))
                    indicator[:, i, j, :] = 1
                    return tntt.TT(
                        tntt.interpolate.dmrg_cross(
                            lambda idxs: self._sample_jacobian(idxs)[i, j, :],
                            [self._I1, self._I2, *self._num_points],
                            eps=self._eps,
                            nswp=50,
                        )
                        .to_ttm()
                        .cores
                        + [
                            tn.ones(self._patch.degree_u + 1).reshape((1, -1, 1, 1)),
                            tn.ones(self._patch.degree_v + 1).reshape((1, -1, 1, 1)),
                            indicator,
                        ],
                        eps=0,
                    )

                i, j = np.meshgrid(np.arange(2), np.arange(2))
                J = sum(
                    np.vectorize(_interp_jacobian, otypes=[tntt.TT])(i, j).flatten()
                ).round(self._eps)

            else:
                # Create TT for each subelement
                i1, i2 = np.meshgrid(np.arange(self._I1), np.arange(self._I2))

                # Sum all subelement TTs and round
                J = sum(map(self._calc_jacobian, i1.flatten(), i2.flatten())).round(
                    self._eps
                )

                # Add cores for non-vanishing basis functions and permute Jacobian
                # matrix to the end
                J = tntt.permute(
                    tntt.TT(
                        J.cores
                        + [
                            tn.ones(self._patch.degree_u + 1).reshape((1, -1, 1, 1)),
                            tn.ones(self._patch.degree_v + 1).reshape((1, -1, 1, 1)),
                        ],
                        eps=0,
                    ),
                    dims=list(range(len(J.cores) - 1))
                    + [len(J.cores) + i - 1 for i in range(1, 3)]
                    + [len(J.cores) - 1],
                    eps=self._eps,
                )

        else:
            # Evaluate Jacobian at each subelement
            J = tn.empty((self._I1, self._I2, *self._num_points, 2, 2))

            for i1, i2 in itertools.product(range(self._I1), range(self._I2)):
                J[i1, i2, ...] = self._calc_jacobian(i1, i2)

        return J

    def _jacobian_det(self):
        """"""
        if self._use_tt:
            if self.interp_jacobian_det:
                # Apply cross approximation
                J_det = tntt.interpolate.dmrg_cross(
                    self._sample_jacobian_det,
                    [self._I1, self._I2, *self._num_points],
                    eps=self._eps,
                    nswp=50,
                ).round(self._eps)

            else:
                # Create TT for each subelement
                i1, i2 = np.meshgrid(np.arange(self._I1), np.arange(self._I2))

                # Vectorize evaluation function and run
                J_det = sum(
                    map(self._calc_jacobian_det, i1.flatten(), i2.flatten())
                ).round(self._eps)

        else:
            # Evaluate Jacobian determinant at each subelement
            J_det = tn.empty((self._I1, self._I2, *self._num_points))

            for i1, i2 in itertools.product(range(self._I1), range(self._I2)):
                J_det[i1, i2, ...] = self._calc_jacobian_det(i1, i2)

        return J_det

    def _basis(self):
        """"""
        if self._use_tt:
            if self.interp_basis:
                raise NotImplementedError()
            else:
                # Evaluate basis and their derivatives at each subelement
                i1, i2 = np.meshgrid(np.arange(self._I1), np.arange(self._I2))
                result = list(map(self._calc_basis, i1.flatten(), i2.flatten()))
                R, dR = sum(x[0] for x in result).round(self._eps), sum(
                    x[1] for x in result
                ).round(self._eps)

        else:
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
        if self._use_tt:
            if self.interp_boundary_jacobian_det:
                # Apply cross approximation and sum boundaries
                boundary_jacobian_func = np.vectorize(
                    lambda i: tntt.TT(
                        [
                            tn.tensor(
                                [1, 0] if (i % 2) == 0 else [0, 1], dtype=tn.float64
                            )
                            .unsqueeze_(0)
                            .unsqueeze_(-1)
                        ]
                        + tntt.interpolate.dmrg_cross(
                            lambda idxs: self._sample_boundary_jacobian_det(i, idxs),
                            [self._I1, self._num_points[0]]
                            if i < 2
                            else [self._I2, self._num_points[1]],
                            eps=self._eps,
                            nswp=50,
                        ).cores,
                        eps=0,
                    ),
                    otypes=[tntt.TT],
                )

                Jx_det, Jy_det = sum(boundary_jacobian_func([0, 1])).round(
                    self._eps
                ), sum(boundary_jacobian_func([2, 3])).round(self._eps)

            else:
                # Evaluate boundary Jacobian
                xboundary_idxs, i1 = np.meshgrid(np.arange(2), np.arange(self._I1))
                yboundary_idxs, i2 = np.meshgrid(np.arange(2, 4), np.arange(self._I2))

                Jx_det, Jy_det = sum(
                    map(
                        self._calc_boundary_jacobian_det,
                        xboundary_idxs.flatten(),
                        i1.flatten(),
                    )
                ).round(self._eps), sum(
                    map(
                        self._calc_boundary_jacobian_det,
                        yboundary_idxs.flatten(),
                        i2.flatten(),
                    )
                ).round(
                    self._eps
                )

        else:
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
        if self._use_tt:
            if self.interp_boundary_basis:
                raise NotImplementedError()

            else:
                # Evaluate boundary Jacobian
                xboundary_idxs, i1 = np.meshgrid(np.arange(2), np.arange(self._I1))
                yboundary_idxs, i2 = np.meshgrid(np.arange(2, 4), np.arange(self._I2))

                Rx, Ry = sum(
                    map(
                        self._calc_boundary_basis,
                        xboundary_idxs.flatten(),
                        i1.flatten(),
                    )
                ).round(self._eps), sum(
                    map(
                        self._calc_boundary_basis,
                        yboundary_idxs.flatten(),
                        i2.flatten(),
                    )
                ).round(
                    self._eps
                )

        else:
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
        if self._use_tt:
            if self.interp_angular:
                raise NotImplementedError()

            else:
                # Evaluate boundary Jacobian
                xboundary_idxs, i1 = np.meshgrid(np.arange(2), np.arange(self._I1))
                yboundary_idxs, i2 = np.meshgrid(np.arange(2, 4), np.arange(self._I2))

                # Sum over boundary subelements
                Ox = sum(
                    map(
                        self._calc_angular,
                        xboundary_idxs.flatten(),
                        itertools.repeat(dir),
                        i1.flatten(),
                    )
                ).round(self._eps)
                Oy = sum(
                    map(
                        self._calc_angular,
                        yboundary_idxs.flatten(),
                        itertools.repeat(dir),
                        i2.flatten(),
                    )
                ).round(self._eps)

                # Permute dimensions
                startx = len(Ox.cores) - 3
                starty = len(Oy.cores) - 3
                Ox = tntt.permute(
                    Ox, np.arange(startx).tolist() + [startx + 1, startx + 2, startx]
                )
                Oy = tntt.permute(
                    Oy, np.arange(starty).tolist() + [starty + 1, starty + 2, starty]
                )

                Ox, Oy = Ox.round(self._eps), Oy.round(self._eps)

        else:
            # Evaluate boundary angular component
            Ox = tn.empty(
                (4, self._ordinates.shape[0], self._num_points[0], self._I1, 2)
            )
            Oy = tn.empty(
                (4, self._ordinates.shape[0], self._num_points[1], self._I2, 2)
            )

            # Calculate and fill
            for boundary_idx, i1 in itertools.product(range(2), range(self._I1)):
                Ox[..., i1, boundary_idx] = self._calc_angular(boundary_idx, dir, i1)
            for boundary_idx, i2 in itertools.product(range(2), range(self._I2)):
                Oy[..., i2, boundary_idx] = self._calc_angular(
                    boundary_idx + 2, dir, i2
                )

            # Permute
            Ox = tn.permute(Ox, [0, 1, 4, 3, 2])
            Oy = tn.permute(Oy, [0, 1, 4, 3, 2])

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
        J = tn.permute(
            self._sample_jacobian(np.array([i1, i2, j1, j2]).T), (2, 0, 1)
        ).reshape((*self._num_points, 2, 2))

        # Return result if not using TT format
        if not self._use_tt:
            return J

        # Convert to TT format
        J = J.reshape((*self._num_points, 2, 1, 1, 2))
        J = tntt.TT(
            J, shape=[(J.shape[i], 1) for i in range(2)] + [(2, 2)], eps=self._eps
        )

        # Apply Kronecker product
        indx = tn.zeros(self._I1).reshape((1, -1, 1, 1))
        indy = tn.zeros(self._I2).reshape((1, -1, 1, 1))
        indx[0, i1[0], 0, 0] = 1
        indy[0, i2[0], 0, 0] = 1

        return tntt.TT([indx, indy] + J.cores, eps=0)

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
        J_det = self._sample_jacobian_det(np.array([i1, i2, j1, j2]).T).reshape(
            self._num_points
        )

        # Return result if not using TT format
        if not self._use_tt:
            return J_det

        # Convert to TT format
        J_det = tntt.TT(J_det, eps=self._eps)

        # Apply Kronecker product
        indx = tn.zeros(self._I1).reshape((1, -1, 1))
        indy = tn.zeros(self._I2).reshape((1, -1, 1))
        indx[0, i1[0], 0] = 1
        indy[0, i2[0], 0] = 1

        return tntt.TT([indx, indy] + J_det.cores, eps=0)

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

        # Return result if not using TT format
        if not self._use_tt:
            return basis_data[..., 0].reshape(
                (*self._num_points, *basis_data.shape[1:-1])
            ), basis_data[..., 1:].reshape(
                (*self._num_points, *basis_data.shape[1:-1], 2)
            )

        # Convert to TT format
        R = tntt.TT(
            basis_data[..., 0].reshape((*self._num_points, *basis_data.shape[1:-1])),
            eps=self._eps,
        )
        dR = tntt.TT(
            basis_data[..., 1:].reshape(
                (*self._num_points, *basis_data.shape[1:-1], 2)
            ),
            eps=self._eps,
        )

        # Apply Kronecker product
        indx = tn.zeros(self._I1).reshape((1, -1, 1))
        indy = tn.zeros(self._I2).reshape((1, -1, 1))
        indx[0, i1[0], 0] = 1
        indy[0, i2[0], 0] = 1

        return tntt.TT([indx, indy] + R.cores, eps=0), tntt.TT(
            [indx, indy] + dR.cores, eps=0
        )

    def _calc_boundary_jacobian_det(self, boundary_idx, i):
        """"""
        param_idx, I0 = (0, self._I1) if boundary_idx < 2 else (1, self._I2)

        # Get indices
        j = np.arange(self._num_points[param_idx])
        i = (i * np.ones(j.size)).astype(int)

        # Calculate Jacobian
        J_det = self._sample_boundary_jacobian_det(boundary_idx, np.array([i, j]).T)

        # Return result if not using TT format
        if not self._use_tt:
            return J_det

        # Convert to TT format
        indbc = tn.zeros(2).reshape((1, -1, 1))
        indxy = tn.zeros(I0).reshape((1, -1, 1))
        indbc[0, boundary_idx % 2, 0] = 1
        indxy[0, i[0], 0] = 1

        return tntt.TT([indbc, indxy, J_det.reshape(1, -1, 1)], eps=0)

    def _calc_boundary_basis(self, boundary_idx, i):
        """"""
        param_idx, I0 = (0, self._I1) if boundary_idx < 2 else (1, self._I2)

        # Get indices
        j = np.arange(self._num_points[param_idx])
        i = (i * np.ones(j.size)).astype(int)

        # Calculate non-vanishing boundary basis functions
        R = self._sample_boundary_basis(boundary_idx, np.array([i, j]).T)

        # Return result if not using TT format
        if not self._use_tt:
            return R

        # Create indicator
        indicator = tn.zeros(I0)
        indicator[i] = 1.0

        return tntt.TT(
            [
                tn.tensor(
                    [1, 0] if boundary_idx % 2 == 0 else [0, 1], dtype=tn.float64
                ).reshape((1, -1, 1)),
                indicator.reshape((1, -1, 1)),
            ]
            + tntt.TT(R, eps=self._eps).cores,
            eps=0,
        )

    def _calc_angular(self, boundary_idx, dir, i):
        """"""
        param_idx, I0 = (0, self._I1) if boundary_idx < 2 else (1, self._I2)

        # Get indices
        j = np.arange(self._num_points[param_idx])
        i = (i * np.ones(j.size)).astype(int)

        # Sample angular component of boundary
        products = self._sample_angular(boundary_idx, dir, np.array([i, j]).T)

        # Return result if not using TT format
        if not self._use_tt:
            return products

        # Create indicator
        ind = tn.zeros(I0)
        ind[i] = 1.0

        return tntt.TT(
            tntt.TT(tn.sqrt(products), eps=self._eps).cores
            + [
                tn.tensor(
                    [1, 0] if boundary_idx % 2 == 0 else [0, 1], dtype=tn.float64
                ).reshape((1, -1, 1)),
                ind.reshape((1, -1, 1)),
            ]
        )

    def _calc_Intg_x(self, local_Intg_x):
        """"""
        # Create TT for each subelement
        i1, i2 = np.meshgrid(np.arange(self._I1), np.arange(self._I2))

        # Sum and round
        return sum(
            map(
                self._get_local_Intg,
                i1.flatten(),
                i2.flatten(),
                itertools.repeat(local_Intg_x),
            )
        ).round(self._eps)

    def _calc_Intg_bound(
        self, boundary_idxs, local_Intg_bound, incident_boundaries=None
    ):
        """"""
        # Get integrals over each boundary
        Intg = [
            sum(
                map(
                    self._get_local_boundary_Intg,
                    itertools.repeat(boundary_idx),
                    np.arange(self._I1 if boundary_idx < 2 else self._I2),
                    itertools.repeat(local_Intg_bound),
                )
            ).round(self._eps)
            for boundary_idx in boundary_idxs
        ]

        # Handle incident boundary conditions
        if incident_boundaries is not None:
            for boundary_idx, boundary in zip(boundary_idxs, incident_boundaries):
                if boundary.from_patch == self._p:
                    # Handle reflective boundary condition
                    # Initialize core
                    quadrant_core = tn.zeros((1, 4, 4, Intg[boundary_idx % 2].R[1]))

                    # Iterate through quadrants
                    for quad in range(4):
                        # Handle reflective boundary condition
                        quadrant = self._quadrants[quad, :].copy()

                        # Calculate normal at axis aligned boundary
                        coords = self._calc_boundary_coords(
                            boundary_idx,
                            self._num_points[0 if boundary_idx < 2 else 1] * [0],
                            list(range(self._num_points[0 if boundary_idx < 2 else 1])),
                        )
                        normals = np.round(self._mesh.normal(self._p, coords)[-1], 10)

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
                        ref_quad = (
                            (self._quadrants == tuple(quadrant))
                            .all(axis=1)
                            .nonzero()[0][0]
                        )

                        # Place quadrant
                        quadrant_core[0, quad, ref_quad, :] = Intg[
                            boundary_idx % 2
                        ].cores[0][0, quad, 0, :]

                    # Replace quadrant core
                    Intg[boundary_idx % 2].set_core(0, quadrant_core)

                else:
                    Intg[boundary_idx % 2] = self.diagonalize(
                        Intg[boundary_idx % 2], [0]
                    )

        return sum(Intg).round(self._eps)

    def _get_local_Intg(self, i1, i2, Intg):
        """"""
        # Index out local integral
        cores = [
            Intg.cores[0][:, [i1], :, :],
            Intg.cores[1][:, [i2], :, :],
        ] + Intg.cores[2:-3]

        # Increase spatial core size to all control points
        xhat = tn.zeros(
            (
                Intg.R[-4],
                self._patch.ctrlpts_size_u,
                self._patch.ctrlpts_size_u,
                Intg.R[-3],
            )
        )
        yhat = tn.zeros(
            (
                Intg.R[-3],
                self._patch.ctrlpts_size_v,
                self._patch.ctrlpts_size_v,
                Intg.R[-2],
            )
        )

        # Place smaller core
        xhat[
            :,
            i1 : i1 + self._patch.degree_u + 1,
            i1 : i1 + self._patch.degree_u + 1,
            :,
        ] = Intg.cores[-3]
        yhat[
            :,
            i2 : i2 + self._patch.degree_v + 1,
            i2 : i2 + self._patch.degree_v + 1,
            :,
        ] = Intg.cores[-2]

        return tntt.TT(cores + [xhat, yhat, Intg.cores[-1]], eps=0).round(self._eps)

    def _get_local_boundary_Intg(self, boundary_idx, i, Intg_bound):
        """"""
        # Index out local integral
        cores = [
            *Intg_bound.cores[:2],
            Intg_bound.cores[2][:, [boundary_idx % 2], :, :],
            Intg_bound.cores[3][:, [i], :, :],
            *Intg_bound.cores[4:-2],
        ]

        # Increase spatial core size to all control points
        xhat = tn.zeros(
            (
                Intg_bound.R[-3],
                self._patch.ctrlpts_size_u,
                self._patch.ctrlpts_size_u,
                Intg_bound.R[-2],
            )
        )
        yhat = tn.zeros(
            (
                Intg_bound.R[-2],
                self._patch.ctrlpts_size_v,
                self._patch.ctrlpts_size_v,
                Intg_bound.R[-1],
            )
        )

        # Place smaller core
        if boundary_idx < 2:
            i1 = i
            i2 = 0 if boundary_idx == 0 else self._I2 - 1
        else:
            i1 = 0 if boundary_idx == 2 else self._I1 - 1
            i2 = i

        xhat[
            :,
            i1 : i1 + self._patch.degree_u + 1,
            i1 : i1 + self._patch.degree_u + 1,
            :,
        ] = Intg_bound.cores[-2]
        yhat[
            :,
            i2 : i2 + self._patch.degree_v + 1,
            i2 : i2 + self._patch.degree_v + 1,
            :,
        ] = Intg_bound.cores[-1]

        return tntt.TT(cores + [xhat, yhat], eps=0).round(self._eps)

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
        products = np.einsum(
            "qj,nj,ji->qni", self._quadrants, self._ordinates[:, 1:], normals
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
    # TT-operations

    @staticmethod
    def diagonalize(tt, core_idxs=None):
        # Get cores
        cores = tt.cores

        # Get cores to diagonalize
        core_idxs = np.arange(len(cores)) if core_idxs is None else core_idxs

        if tt.is_ttm:
            for i in core_idxs:
                assert cores[i].shape[2] == 1
                cores[i].squeeze_(2)

        # Iterate through cores
        cores = [
            (
                tn.einsum("ijk,jm->ijmk", c, tn.eye(c.shape[1]))
                if i in core_idxs
                else (c.unsqueeze_(2) if not tt.is_ttm else c)
            )
            for i, c in enumerate(cores)
        ]

        # Create new tt and round
        return tntt.TT(cores, eps=0)

    # ========================================================================
    # I/O

    def _print_tt(self, name):
        """"""
        print(
            "{:15s} {:25s} {:10.2f}  {:10.2f}".format(
                name,
                ",".join(map(str, self._assembly_info[name]["ranks"])),
                self._assembly_info[name]["compression"],
                self._assembly_info[name]["elapsed time"],
            )
        )

    def save_tt_info(self, path, round_data=True):
        """"""
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
    # Other

    def _append_tt_info(self, name, tt):
        """"""
        entries = sum([tn.numel(c) for c in tt.cores])
        self._assembly_info[name] = {
            "shape": tt.shape,
            "ranks": tt.R[1:-1],
            "entries": entries,
            "compression": (
                np.prod(tt.N) if not tt.is_ttm else np.prod(tt.N) * np.prod(tt.M)
            )
            / entries,
            "elapsed time": time.time() - self._start_time,
        }

        # Print info if verbose is on
        if self._verbose:
            self._print_tt(name)

    # ========================================================================
    # Getters

    @property
    def Int_N(self):
        """"""
        Int_N = tntt.TT(
            [
                tn.ones(self._quadrants.shape[0]).reshape((1, 1, -1, 1)),
                tn.tensor(self._ordinates[:, 0]).reshape((1, 1, -1, 1)),
                tn.eye(self._xs_server.num_groups).unsqueeze_(0).unsqueeze_(-1),
                tn.eye(self._patch.ctrlpts_size_u).unsqueeze_(0).unsqueeze_(-1),
                tn.eye(self._patch.ctrlpts_size_v).unsqueeze_(0).unsqueeze_(-1),
            ]
        )
        Int_N.reduce_dims()
        return Int_N

    @property
    def Int_I(self):
        # Create integral operator
        Int_I = tntt.ones(
            [
                self._quadrants.shape[0],
                self._ordinates.shape[0],
                self._xs_server.num_groups,
                self._patch.ctrlpts_size_u,
                self._patch.ctrlpts_size_v,
            ]
        )
        Int_I.reduce_dims()
        return Int_I.to_ttm().t()
