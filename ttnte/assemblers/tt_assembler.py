import itertools
import time
from typing import Optional, Tuple

import cotengra as ctg
import numpy as np
import pandas as pd
import torch as tn
import torchtt as tntt

from ttnte.iga import IGAMesh
from ttnte.xs import Server

from .matrix_assembler import MatrixAssembler


class TTAssembler(MatrixAssembler):
    """
    IGA assembler class for operators in the TT format.

    .. note::
        The TT-cross approximation is only useful if the resulting
        TT will be compressed.

    Attributes
    ----------
    interp_jacobian: bool, default=True
        Use TT-coss to build Jacobian operator.
    interp_jacobian_det: bool, default=True
        Use TT-cross to build boundary Jacobian operator.
    interp_basis: bool, default=False
        Use TT-cross to compute basis evaluations TT.
    interp_boundary_jacobian_det: bool, default=True
        Use TT-cross to compute the Jacobian determinant operator
        for the boundaries.
    interp_boundary_basis: bool, default=False
        Use TT-cross to compute the basis at the boundary.
    interp_angular: bool, default=False
        Use TT-cross to compute the angular component of the boundaries.
    """

    # Assembly settings
    interp_jacobian = True
    interp_jacobian_det = True
    interp_basis = False

    interp_boundary_jacobian_det = True
    interp_boundary_basis = False
    interp_angular = False

    def __init__(
        self,
        mesh: IGAMesh,
        xs_server: Server,
        num_ordinates: int,
        num_points: Optional[Tuple[int]] = None,
    ):
        """
        Initialize TTAssembler.

        Parameters
        ----------
        mesh: ttnte.iga.IGAMesh
            IGA mesh.
        xs_server: ttnte.xs.Server
            XS server.
        num_ordinates: int
            Total number of ordinates.
        num_points: tuple of int or None
            Size of spatial quadrature. If ``num_points is None`` then the
            spatial quadrature for each knot span is :math:`(p + 1, q + 1)`
            where :math:`p` and :math:`q` is the degree of the NURBS basis
            functions.
        """
        # Initialize base class
        super().__init__(
            mesh=mesh,
            xs_server=xs_server,
            num_ordinates=num_ordinates,
            num_points=num_points,
        )

    # ========================================================================
    # Main build methods

    def build(
        self,
        eps: float = 1e-10,
        verbose: bool = True,
        use_tt: bool = True,
    ):
        """
        Build TT operators.

        Parameters
        ----------
        eps: float, default=1e-10
            Threshold for TT decomposition.
        verbose: bool, default=True
            Print progress.
        use_tt: bool, default=True
            Use TT-cross or concatenation to build the operators.

        Returns
        -------
        H: torch.TT
            Loss operator.
        S: torch.TT
            Scattering operator.
        F: torch.TT
            Fission operator.
        B_in: torch.TT
            Incident boundary operator.
        B_out: torch.TT
            Outgoing boundary operator.
        """
        self._eps = eps
        self._use_tt = use_tt

        # Begin build process
        self._initialize_build(verbose)

        # Setup spherical harmonics
        self._setup_sph_harm()

        # Create local operators for each patch
        operators = list(map(self._build_patch, np.arange(len(self._mesh.patches))))

        # Concatenate operators
        H = sum([o[0] for o in operators]).round(self._eps)
        S = sum([o[1] for o in operators]).round(self._eps)
        F = sum([o[2] for o in operators]).round(self._eps)
        B_in = sum([o[3] for o in operators]).round(self._eps)
        B_out = sum([o[4] for o in operators]).round(self._eps)
        del operators

        # Diagonalize direction, energy, and patch dimensions of loss operator
        H = self._diagonalize(H, np.arange(5))
        B_out = self._diagonalize(B_out, np.arange(5))

        # Diagonalize patch dimension in fission operator
        F = self._diagonalize(F, [4])
        S = self._diagonalize(S, [4])

        # Reduce dimensions
        H.reduce_dims()
        S.reduce_dims()
        F.reduce_dims()
        B_in.reduce_dims()
        B_out.reduce_dims()

        # Round
        H = H.round(self._eps)
        S = S.round(self._eps)
        F = F.round(self._eps)
        B_in = B_in.round(self._eps)
        B_out = B_out.round(self._eps)

        # Save TT info
        if self._verbose:
            print("\nFinal Operators")
        self._p = -1
        self._append_tt_info("H", H)
        self._append_tt_info("S", S)
        self._append_tt_info("F", F)
        self._append_tt_info("B_in", B_in)
        self._append_tt_info("B_out", B_out)

        return H, S, F, B_in, B_out

    def _build_patch(self, p):
        """
        Build TT operator for a patch.

        Parameters
        ----------
        p: int
            Patch index.

        Returns
        -------
        H: torch.TT
            Local loss operator.
        S: torch.TT
            Local scattering operator.
        F: torch.TT
            Local fission operator.
        B_in: torch.TT
            Local incident boundary operator.
        B_out: torch.TT
            Local outgoing boundary operator.
        """
        # Setup current patch
        self._setup_current_patch(p)

        if self._use_tt:
            # Get Jacobian
            J = self._jacobian()
            self._append_tt_info("J", J)

            # Cross-interpolate Jacobian determinant
            J_det = self._jacobian_det()
            self._append_tt_info("J_det", J_det)

            # Calculate basis data at quadrature points for each knot span
            R, dR = self._basis()
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
                self._diagonalize(J, np.arange(len(J.cores) - 1))
                .fast_matvec(dR, self._eps)
                .round(self._eps)
            )
            del J, dR
            self._append_tt_info("JdR", JdR)
            JdR = self._diagonalize(JdR, [0, 1, len(JdR.cores) - 1])

            # Transpose quadrature dimensions for sum
            R = self._diagonalize(R, [0, 1])
            R.set_core(2, tn.permute(R.cores[2], (0, 2, 1, 3)))
            R.set_core(3, tn.permute(R.cores[3], (0, 2, 1, 3)))
            JdR.set_core(2, tn.permute(JdR.cores[2], (0, 2, 1, 3)))
            JdR.set_core(3, tn.permute(JdR.cores[3], (0, 2, 1, 3)))

            # Calculate local volume integrals
            Intg_int = self._concat_integrals(
                tntt.TT(
                    tntt.amen_mm(R, J_detRT, nswp=50, eps=self._eps)
                    .round(self._eps)
                    .cores
                    + [tn.ones(1).reshape((1, 1, 1, 1))]
                )
            )
            Intg_str = self._concat_integrals(
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
            # Get Jacobian
            J = super()._jacobian()

            # Cross-interpolate Jacobian determinant
            J_det = super()._jacobian_det()

            # Calculate basis data at quadrature points for each knot span
            R, dR = super()._basis()

            # Build local volume integrals
            Intg = super()._build_local_integrals(J, J_det, R, dR)
            del J, J_det, R, dR

            # Build TTs
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

        # Create patch indicator
        patch_ind = tn.zeros(self._mesh.num_patches).reshape((1, -1, 1, 1))
        patch_ind[0, self._p, ...] = 1

        # Build operators with remaining cores
        self._append_tt_info("Intg_int", Intg_int)
        self._append_tt_info("Intg_str", Intg_str)
        H = self._build_loss(Intg_int, Intg_str, patch_ind)
        self._append_tt_info("H", H)
        S = self._build_scatter(Intg_int, patch_ind)
        self._append_tt_info("S", S)
        F = self._build_fission(Intg_int, patch_ind)
        self._append_tt_info("F", F)
        B_in = self._build_incident_boundary()
        self._append_tt_info("B_in", B_in)
        B_out = self._build_outgoing_boundary(patch_ind)
        self._append_tt_info("B_out", B_out)

        return H, S, F, B_in, B_out

    def _build_loss(self, Intg_int, Intg_str, patch_ind):
        """"""
        # Add angular and energy dimensions
        H = (
            tntt.TT(
                [
                    tn.ones(self._quadrants.shape[0]).reshape((1, -1, 1, 1)),
                    tn.ones(self._ordinates[0].shape[0]).reshape((1, -1, 1, 1)),
                    tn.ones(self._ordinates[1].shape[0]).reshape((1, -1, 1, 1)),
                    tn.tensor(self._xs_server.total(self._patch.name)).reshape(
                        (1, -1, 1, 1)
                    ),
                    patch_ind,
                ]
                + Intg_int.cores
                + [tn.ones(1).reshape((1, 1, 1, 1))],
                eps=0,
            )
            - tntt.TT(
                [
                    tn.tensor(self._quadrants[:, 0]).reshape((1, -1, 1, 1)),
                    tn.tensor(self._ordinates[0][:, 1]).reshape((1, -1, 1, 1)),
                    tn.ones(self._ordinates[1].shape[0]).reshape((1, -1, 1, 1)),
                    tn.ones(self._xs_server.num_groups).reshape((1, -1, 1, 1)),
                    patch_ind,
                ]
                + Intg_str.cores[:-1]
                + [Intg_str.cores[-1][:, [0], :, :]],
                eps=0,
            )
            - tntt.TT(
                [
                    tn.tensor(self._quadrants[:, 1]).reshape((1, -1, 1, 1)),
                    tn.tensor(np.sqrt(1 - self._ordinates[0][:, 1] ** 2)).reshape(
                        (1, -1, 1, 1)
                    ),
                    tn.tensor(np.cos(self._ordinates[1][:, 1])).reshape((1, -1, 1, 1)),
                    tn.ones(self._xs_server.num_groups).reshape((1, -1, 1, 1)),
                    patch_ind,
                ]
                + Intg_str.cores[:-1]
                + [Intg_str.cores[-1][:, [1], :, :]],
                eps=0,
            )
        )

        # Remove single size dimensions
        H.reduce_dims(exclude=np.arange(len(H.cores) - 1))
        return H

    def _build_scatter(self, Intg_int, patch_ind):
        """"""
        # Angular and energy component of scattering
        S = (
            self._Y
            @ tntt.TT(
                tntt.eye(
                    [
                        self._quadrants.shape[0],
                        self._ordinates[0].shape[0],
                        self._ordinates[1].shape[0],
                    ]
                ).cores
                + tntt.TT(
                    tn.tensor(
                        self._xs_server.scatter_gtg(self._patch.name)[
                            :, :, np.newaxis, :
                        ]
                    ),
                    shape=[
                        (self._xs_server.num_moments, 1),
                        (self._xs_server.num_groups, self._xs_server.num_groups),
                    ],
                    eps=self._eps,
                ).cores
            )
        ).round(self._eps)
        S.reduce_dims(exclude=[0, 1, 2, 4])

        # Apply angular integration
        S.cores[-3] *= tn.tensor(self._ordinates[0][:, 0]).reshape((1, 1, -1, 1))
        S.cores[-2] *= tn.tensor(self._ordinates[1][:, 0]).reshape((1, 1, -1, 1))

        # Add spatial integral
        S = tntt.TT(S.cores + [patch_ind] + Intg_int.cores)
        return S

    def _build_fission(self, Intg_int, patch_ind):
        """"""
        # Fission operator
        F = tntt.TT(
            [
                tn.ones(2 * [self._quadrants.shape[0]]).unsqueeze_(0).unsqueeze_(-1),
                tn.outer(
                    tn.ones(self._ordinates[0].shape[0]),
                    tn.tensor(self._ordinates[0][:, 0]),
                )
                .unsqueeze_(0)
                .unsqueeze_(-1),
                tn.outer(
                    tn.ones(self._ordinates[1].shape[0]),
                    tn.tensor(self._ordinates[1][:, 0]),
                )
                .unsqueeze_(0)
                .unsqueeze_(-1),
                tn.outer(
                    tn.tensor(self._xs_server.chi),
                    tn.tensor(self._xs_server.nu_fission(self._patch.name)),
                )
                .unsqueeze_(0)
                .unsqueeze_(-1),
                patch_ind,
            ]
            + Intg_int.cores,
            eps=0,
        ).round(self._eps)
        F.reduce_dims(exclude=np.arange(len(F.cores) - 1))
        return F

    def _build_outgoing_boundary(self, patch_ind):
        """"""
        if self._use_tt:
            # Compute Jacobian of the boundary integral
            Jx_det, Jy_det = self._boundary_jacobian_det()
            self._append_tt_info("Jx_det_out", Jx_det)
            self._append_tt_info("Jy_det_out", Jy_det)

            # Get basis data
            Rx, Ry = self._boundary_basis()
            self._append_tt_info("Rx_out", Rx)
            self._append_tt_info("Ry_out", Ry)

            # Get angular data
            Ox, Oy = self._angular(dir=1.0)
            self._append_tt_info("Ox_out", Oy)
            self._append_tt_info("Oy_out", Oy)

            # Get local boundary integrals
            Intg = self._build_boundary_integrals(
                (Jx_det, Jy_det), (Rx, Ry), (Ox, Oy), "out"
            )
            Intg.reduce_dims()
            del Jx_det, Jy_det, Rx, Ry, Ox, Oy

        else:
            # Compute Jacobian of the boundary integral
            Jx_det, Jy_det = super()._boundary_jacobian_det()

            # Get basis data
            Rx, Ry = super()._boundary_basis()

            # Get angular data
            Ox, Oy = super()._angular(dir=1.0)

            # Get local boundary
            raise NotImplementedError()

        # Save boundary integral TT info
        self._append_tt_info("Intg_bound_out", Intg)

        # Append energy identity matrix and permute TT
        H_bound = tntt.permute(
            tntt.TT(
                [tn.ones(self._xs_server.num_groups).reshape((1, -1, 1, 1)), patch_ind]
                + Intg.cores,
                eps=0,
            ),
            [2, 3, 4, 0, 1, 5, 6],
            eps=self._eps,
        ).round(self._eps)
        self._append_tt_info("H_bound_out", H_bound)
        return H_bound

    def _build_incident_boundary(self):
        """"""
        c = self if self._use_tt else super()
        # Compute Jacobian of the boundary integral
        Jx_det, Jy_det = c._boundary_jacobian_det()

        # Get basis data
        Rx, Ry = c._boundary_basis()

        # Get angular data
        Ox, Oy = c._angular(dir=-1.0)

        if self._use_tt:
            for boundary_idx in range(2):
                # Get adjacent patch index
                xp = self._mesh.get_connected_patch(self._p, coord=(0.5, boundary_idx))
                yp = self._mesh.get_connected_patch(self._p, coord=(boundary_idx, 0.5))

                # Make mask
                mask = tn.tensor([0, 1] if boundary_idx == 0 else [1, 0]).reshape(
                    (1, -1, 1)
                )

                if xp is None:
                    # Apply mask
                    Jx_det.set_core(0, Jx_det.cores[0].clone() * mask)
                    Rx.set_core(0, Rx.cores[0].clone() * mask)
                    Ox.set_core(3, Ox.cores[3].clone() * mask)

                if yp is None:
                    # Apply mask
                    Jy_det.set_core(0, Jy_det.cores[0].clone() * mask)
                    Ry.set_core(0, Ry.cores[0].clone() * mask)
                    Oy.set_core(3, Oy.cores[3].clone() * mask)

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

            # Get local boundary integrals
            Intg = self._build_boundary_integrals(
                (Jx_det, Jy_det), (Rx, Ry), (Ox, Oy), "in"
            )
            Intg.reduce_dims(exclude=[0, 1, 2, 3, 7, 8])

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
            [2, 3, 4, 0, 1, 5, 6],
            eps=self._eps,
        ).round(self._eps)
        self._append_tt_info("H_bound_in", H_bound)
        return self._diagonalize(H_bound, [1, 2])

    def _build_boundary_integrals(self, J_det, R, Or, tag):
        """"""
        Jx_det, Jy_det = J_det
        Rx, Ry = R
        Ox, Oy = Or

        # Add angular component to basis
        Rx = tntt.TT(
            [
                tn.ones(self._quadrants.shape[0]).reshape((1, -1, 1)),
                tn.ones(self._ordinates[0].shape[0]).reshape((1, -1, 1)),
                tn.ones(self._ordinates[1].shape[0]).reshape((1, -1, 1)),
            ]
            + Rx.cores,
            eps=0,
        )
        Ry = tntt.TT(
            [
                tn.ones(self._quadrants.shape[0]).reshape((1, -1, 1)),
                tn.ones(self._ordinates[0].shape[0]).reshape((1, -1, 1)),
                tn.ones(self._ordinates[1].shape[0]).reshape((1, -1, 1)),
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
                tn.ones(self._ordinates[0].shape[0]).reshape((1, -1, 1)),
                tn.ones(self._ordinates[1].shape[0]).reshape((1, -1, 1)),
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
                tn.ones(self._ordinates[0].shape[0]).reshape((1, -1, 1)),
                tn.ones(self._ordinates[1].shape[0]).reshape((1, -1, 1)),
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
        ORJx_det = self._diagonalize(ORJx_det, np.arange(len(ORJx_det.cores) - 3))
        ORJy_det = self._diagonalize(ORJy_det, np.arange(len(ORJy_det.cores) - 3))

        # Transpose quadrature for local sum
        ORJx_det.set_core(5, tn.permute(ORJx_det.cores[5], (0, 2, 1, 3)))
        ORJy_det.set_core(5, tn.permute(ORJy_det.cores[5], (0, 2, 1, 3)))

        # Transpose basis functions
        ORTx = ORx.to_ttm()
        ORTx.set_core(6, tn.permute(ORTx.cores[6], (0, 2, 1, 3)))
        ORTx.set_core(7, tn.permute(ORTx.cores[7], (0, 2, 1, 3)))
        ORTy = ORy.to_ttm()
        ORTy.set_core(6, tn.permute(ORTy.cores[6], (0, 2, 1, 3)))
        ORTy.set_core(7, tn.permute(ORTy.cores[7], (0, 2, 1, 3)))
        del ORx, ORy

        # Calculate local surface integral
        local_Intg_x = tntt.amen_mm(ORJx_det, ORTx, nswp=50, eps=self._eps).round(
            self._eps
        )
        local_Intg_y = tntt.amen_mm(ORJy_det, ORTy, nswp=50, eps=self._eps).round(
            self._eps
        )
        Intg = (
            self._concat_boundary_integrals(
                [0, 1],
                local_Intg_x,
                connected_patches=(
                    [
                        self._mesh.get_connected_patch(self._p, (0.5, i))
                        for i in range(2)
                    ]
                    if tag == "in"
                    else None
                ),
            )
            + self._concat_boundary_integrals(
                [2, 3],
                local_Intg_y,
                connected_patches=(
                    [
                        self._mesh.get_connected_patch(self._p, (i, 0.5))
                        for i in range(2)
                    ]
                    if tag == "in"
                    else None
                ),
            )
        ).round(self._eps)

        return Intg

    def _setup_sph_harm(self):
        """"""
        self._Y = []

        for n in range(self._xs_server.num_moments):
            # Build spherical harmonics at l
            Yl = tntt.TT(self._sph_harm(n), eps=self._eps).to_ttm()

            # Create transpose over ordinate
            YlT = Yl.clone()
            YlT.set_core(1, tn.permute(YlT.cores[1], (0, 2, 1, 3)))
            YlT.set_core(2, tn.permute(YlT.cores[2], (0, 2, 1, 3)))
            YlT.set_core(3, tn.permute(YlT.cores[3], (0, 2, 1, 3)))

            # Diagonalize m and quadrant
            Yl = self._diagonalize(Yl, [0])

            # Compute outer product
            Yl = Yl @ YlT

            # Take Hadamard product with angular integral and sum each moment
            ind = tn.zeros((1, self._xs_server.num_moments))
            ind[0, n] = 1
            self._Y.append(
                tntt.TT(
                    (
                        tntt.TT(
                            [
                                tn.ones((1, int(1 + n))).unsqueeze_(0).unsqueeze_(-1),
                                tn.eye(self._quadrants.shape[0])
                                .unsqueeze_(0)
                                .unsqueeze_(-1),
                                tn.eye(self._ordinates[0].shape[0])
                                .unsqueeze_(0)
                                .unsqueeze_(-1),
                                tn.eye(self._ordinates[1].shape[0])
                                .unsqueeze_(0)
                                .unsqueeze_(-1),
                            ]
                        )
                        @ Yl
                    ).cores
                    + [ind.unsqueeze_(0).unsqueeze_(-1)],
                    eps=0,
                ),
            )
            self._Y[-1].reduce_dims([1, 2, 3, 4])

        # Add all moments
        self._Y = sum(self._Y).round(self._eps)

        # Add energy groups core
        self._Y = tntt.TT(
            self._Y.cores
            + [tn.eye(self._xs_server.num_groups).unsqueeze_(0).unsqueeze_(-1)],
            eps=0,
        )

    # ========================================================================
    # Assembly steps
    def _jacobian(self):
        """"""
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
            return sum(
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
            return tntt.permute(
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

    def _jacobian_det(self):
        """"""
        if self.interp_jacobian_det:
            # Apply cross approximation
            return tntt.interpolate.dmrg_cross(
                self._sample_jacobian_det,
                [self._I1, self._I2, *self._num_points],
                eps=self._eps,
                nswp=50,
            ).round(self._eps)

        else:
            # Create TT for each subelement
            i1, i2 = np.meshgrid(np.arange(self._I1), np.arange(self._I2))

            # Vectorize evaluation function and run
            return sum(map(self._calc_jacobian_det, i1.flatten(), i2.flatten())).round(
                self._eps
            )

    def _basis(self):
        """"""
        if self.interp_basis:
            raise NotImplementedError()
        else:
            # Evaluate basis and their derivatives at each subelement
            i1, i2 = np.meshgrid(np.arange(self._I1), np.arange(self._I2))
            result = list(map(self._calc_basis, i1.flatten(), i2.flatten()))
            return sum(x[0] for x in result).round(self._eps), sum(
                x[1] for x in result
            ).round(self._eps)

    def _boundary_jacobian_det(self):
        """"""
        if self.interp_boundary_jacobian_det:
            # Apply cross approximation and sum boundaries
            boundary_jacobian_func = np.vectorize(
                lambda i: tntt.TT(
                    [
                        tn.tensor([1, 0] if (i % 2) == 0 else [0, 1], dtype=tn.float64)
                        .unsqueeze_(0)
                        .unsqueeze_(-1)
                    ]
                    + tntt.interpolate.dmrg_cross(
                        lambda idxs: self._sample_boundary_jacobian_det(i, idxs),
                        (
                            [self._I1, self._num_points[0]]
                            if i < 2
                            else [self._I2, self._num_points[1]]
                        ),
                        eps=self._eps,
                        nswp=50,
                    ).cores,
                    eps=0,
                ),
                otypes=[tntt.TT],
            )

            return sum(boundary_jacobian_func([0, 1])).round(self._eps), sum(
                boundary_jacobian_func([2, 3])
            ).round(self._eps)

        else:
            # Evaluate boundary Jacobian
            xboundary_idxs, i1 = np.meshgrid(np.arange(2), np.arange(self._I1))
            yboundary_idxs, i2 = np.meshgrid(np.arange(2, 4), np.arange(self._I2))

            return sum(
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

    def _boundary_basis(self):
        """"""
        if self.interp_boundary_basis:
            raise NotImplementedError()

        else:
            # Evaluate boundary Jacobian
            xboundary_idxs, i1 = np.meshgrid(np.arange(2), np.arange(self._I1))
            yboundary_idxs, i2 = np.meshgrid(np.arange(2, 4), np.arange(self._I2))

            return sum(
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

    def _angular(self, dir):
        """"""
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

            return Ox.round(self._eps), Oy.round(self._eps)

    # ========================================================================
    # Non-interpolation methods

    def _calc_jacobian(self, i1, i2):
        """"""
        J = super()._calc_jacobian(i1, i2)

        # Convert to TT format
        J = J.reshape((*self._num_points, 2, 1, 1, 2))
        J = tntt.TT(
            J, shape=[(J.shape[i], 1) for i in range(2)] + [(2, 2)], eps=self._eps
        )

        # Apply Kronecker product
        indx = tn.zeros(self._I1).reshape((1, -1, 1, 1))
        indy = tn.zeros(self._I2).reshape((1, -1, 1, 1))
        indx[0, i1, 0, 0] = 1
        indy[0, i2, 0, 0] = 1

        return tntt.TT([indx, indy] + J.cores, eps=0)

    def _calc_jacobian_det(self, i1, i2):
        """"""
        J_det = super()._calc_jacobian_det(i1, i2)

        # Convert to TT format
        J_det = tntt.TT(J_det, eps=self._eps)

        # Apply Kronecker product
        indx = tn.zeros(self._I1).reshape((1, -1, 1))
        indy = tn.zeros(self._I2).reshape((1, -1, 1))
        indx[0, i1, 0] = 1
        indy[0, i2, 0] = 1

        return tntt.TT([indx, indy] + J_det.cores, eps=0)

    def _calc_basis(self, i1, i2):
        """"""
        R, dR = super()._calc_basis(i1, i2)

        # Convert to TT format
        R = tntt.TT(R, eps=self._eps)
        dR = tntt.TT(dR, eps=self._eps)

        # Apply Kronecker product
        indx = tn.zeros(self._I1).reshape((1, -1, 1))
        indy = tn.zeros(self._I2).reshape((1, -1, 1))
        indx[0, i1, 0] = 1
        indy[0, i2, 0] = 1

        return tntt.TT([indx, indy] + R.cores, eps=0), tntt.TT(
            [indx, indy] + dR.cores, eps=0
        )

    def _calc_boundary_jacobian_det(self, boundary_idx, i):
        """"""
        J_det = super()._calc_boundary_jacobian_det(boundary_idx, i)

        # Convert to TT format
        indbc = tn.zeros(2).reshape((1, -1, 1))
        indxy = tn.zeros(self._I1 if boundary_idx < 2 else self._I2).reshape((1, -1, 1))
        indbc[0, boundary_idx % 2, 0] = 1
        indxy[0, i, 0] = 1

        return tntt.TT([indbc, indxy, J_det.reshape(1, -1, 1)], eps=0)

    def _calc_boundary_basis(self, boundary_idx, i):
        """"""
        R = super()._calc_boundary_basis(boundary_idx, i)

        # Create indicator
        indicator = tn.zeros(self._I1 if boundary_idx < 2 else self._I2)
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
        products = super()._calc_angular(boundary_idx, dir, i)

        # Create indicator
        ind = tn.zeros(self._I1 if boundary_idx < 2 else self._I2)
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

    def _concat_integrals(self, local_Intg):
        """"""
        # Create TT for each subelement
        i1, i2 = np.meshgrid(np.arange(self._I1), np.arange(self._I2))

        # Sum and round
        return sum(
            map(
                self._get_local_Intg,
                i1.flatten(),
                i2.flatten(),
                itertools.repeat(local_Intg),
            )
        ).round(self._eps)

    def _concat_boundary_integrals(
        self, boundary_idxs, local_Intg_bound, connected_patches=None
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
        if connected_patches is not None:
            for boundary_idx, p in zip(boundary_idxs, connected_patches):
                # Patch indicator
                patch_ind = tn.zeros(
                    (1, self._mesh.num_patches, self._mesh.num_patches, 1)
                )

                if p == self._p:
                    patch_ind[0, self._p, self._p, 0] = 1

                    # Handle reflective boundary condition
                    # Initialize core
                    quadrant_core = tn.zeros((1, 4, 4, Intg[boundary_idx % 2].R[1]))

                    # Iterate through quadrants
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
                    if p is not None:
                        # Indicate connected patch
                        patch_ind[0, self._p, p, 0] = 1

                        # Reflect across opposite axis core
                        Intg[boundary_idx % 2].set_core(
                            len(Intg[boundary_idx % 2].cores)
                            - (1 if boundary_idx < 2 else 2),
                            tn.flip(
                                Intg[boundary_idx % 2].cores[
                                    -1 if boundary_idx < 2 else -2
                                ],
                                dims=[2],
                            ),
                        )

                    # Handle vacuum boundary condition
                    Intg[boundary_idx % 2] = self._diagonalize(
                        Intg[boundary_idx % 2], [0]
                    )

                # Append patch core
                Intg[boundary_idx % 2] = tntt.TT(
                    [patch_ind] + Intg[boundary_idx % 2].cores, eps=0
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
            *Intg_bound.cores[:3],
            Intg_bound.cores[3][:, [boundary_idx % 2], :, :],
            Intg_bound.cores[4][:, [i], :, :],
            *Intg_bound.cores[5:-2],
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
    # I/O

    def _print_patch(self, p):
        """"""
        if self._verbose:
            print(f"Assembling Patch {p + 1}")
            print(
                "{:15s} {:25s} {:10s}  {:15s}".format(
                    "Step", "Ranks", "Compression", "Elapsed Time (s)"
                )
            )

    def _print_tt(self, name):
        """"""
        print(
            "{:15s} {:25s} {:10.2f}  {:10.2f}".format(
                name,
                ",".join(map(str, self._assembly_info[(name, self._p)]["ranks"])),
                self._assembly_info[(name, self._p)]["compression"],
                self._assembly_info[(name, self._p)]["elapsed time"],
            )
        )

    def save_tt_info(self, path, round_data=True):
        """
        Save TT info including shape, ranks, number of entries, compression, and elapsed
        time.

        Parameters
        ----------
        path: str
            Path to save CSV file.
        round_data: bool, default=True
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
    # Other

    def _append_tt_info(self, name, tt):
        """"""
        entries = sum([tn.numel(c) for c in tt.cores])
        self._assembly_info[(name, self._p)] = {
            "shape": tt.shape,
            "ranks": tt.R[1:-1],
            "entries": entries,
            "compression": self.compression(tt),
            "elapsed time": time.time() - self._start_time,
        }

        # Print info if verbose is on
        if self._verbose:
            self._print_tt(name)

    def angular_integral(self, psi: tn.Tensor):
        """"""
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
    # TT methods

    @staticmethod
    def _diagonalize(tt, core_idxs=None):
        """"""
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
                ctg.einsum("ijk,jm->ijmk", c, tn.eye(c.shape[1]))
                if i in core_idxs
                else (c.unsqueeze_(2) if not tt.is_ttm else c)
            )
            for i, c in enumerate(cores)
        ]

        # Create new tt and round
        return tntt.TT(cores, eps=0)

    def combine(self, tt, core_idxs):
        """
        Combine TT cores within TT.

        Parameters
        ----------
        tt: torch.TT
            TT to combine some cores.
        core_idxs: list of ints
            Core indices to combine. Indices must be consecutive.

        Returns
        -------
        tt_out: torch.TT
            Resulting TT.
        """
        # Get cores
        cores = tt.cores

        # Get each side of cores
        left_set = cores[: core_idxs[0]]
        right_set = cores[core_idxs[-1] + 1 :]

        # Contract
        super_core = cores[core_idxs[0]]
        if tt.is_ttm:
            for i in core_idxs[1:]:
                super_core = ctg.einsum("abcd,dijk->abicjk", (super_core, cores[i]))
                super_core = super_core.reshape(
                    (
                        super_core.shape[0],
                        np.prod(super_core.shape[1:3]),
                        np.prod(super_core.shape[3:5]),
                        super_core.shape[-1],
                    )
                )

        else:
            for i in core_idxs[1:]:
                super_core = ctg.einsum("abc,cij->abij", (super_core, cores[i]))
                super_core = super_core.reshape(
                    (
                        super_core.shape[0],
                        np.prod(super_core.shape[1:3]),
                        super_core[-1],
                    )
                )

        return tntt.TT(left_set + [super_core] + right_set, eps=0).round(self._eps)

    @staticmethod
    def compression(A: tntt.TT):
        """
        Compute compression of TT.

        .. math::
            Compression = \\frac{Number of elements in the full tensor}\
            {Number of elements in the TT}

        Parameters
        ----------
        A: torch.TT
            TT.

        Returns
        -------
        value: float
            Compression of TT format.
        """
        return (np.prod(A.N) if not A.is_ttm else np.prod(A.N) * np.prod(A.M)) / sum(
            [tn.numel(c) for c in A.cores]
        )
