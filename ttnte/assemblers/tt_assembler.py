import itertools
import multiprocessing as mp
import time
from typing import Optional, Tuple

import cotengra as ctg
import numpy as np
import torch as tn
import torchtt as tntt

from ttnte.__init__ import IS_NOTEBOOK
from ttnte.assemblers.matrix_assembler import MatrixAssembler, Operators, PatchInfo
from ttnte.cad.patch import Patch
from ttnte.iga import IGAMesh
from ttnte.xs import Server

if IS_NOTEBOOK:
    import tqdm.notebook as tqdm
else:
    import tqdm


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
    interp_jacobian = False
    interp_jacobian_det = False
    interp_basis = False

    interp_boundary_jacobian_det = False
    interp_boundary_basis = False
    interp_angular = False

    def __init__(
        self,
        mesh: IGAMesh,
        xs_server: Server,
        num_ordinates: int,
        num_points: Optional[Tuple[int]] = None,
        max_processes: int = max(1, mp.cpu_count() - 1),
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
        max_processes: int, default=max(1, multiprocessing.cpu_count() - 1)
            Maximum allowed processes.
        """
        # Initialize base class
        super().__init__(
            mesh=mesh,
            xs_server=xs_server,
            num_ordinates=num_ordinates,
            num_points=num_points,
            max_processes=max_processes,
        )

    # ========================================================================
    # Main build methods

    def build(
        self,
        eps: float = 1e-10,
        verbose: bool = True,
        use_tt: bool = True,
        **kwargs,
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
        self._eps = eps
        self._use_tt = use_tt
        ops = {name: op for name, op in self._build(kwargs, verbose).items()}

        # Diagonalize
        for name in ["H", "B_out"]:
            if name in ops:
                ops[name] = self._diagonalize(ops[name], np.arange(5))
        for name in ["S", "F"]:
            if name in ops:
                ops[name] = self._diagonalize(ops[name], [4])

        # Reduce dims and final rounding
        for name, op in ops.items():
            if name != "q":
                op.reduce_dims()
                ops[name] = op.round(self._eps)

        # Print final operators
        self._print_final(ops)

        return Operators(**ops)

    def _build_local_integrals(self, pinfo: PatchInfo, R, JRT, J=None, dR=None):
        if not self._use_tt:
            Intg_int, Intg_str = super()._build_local_integrals(pinfo, R, JRT, J, dR)

            pidx = tn.zeros(self._mesh.num_patches).reshape((1, -1, 1, 1))
            pidx[0, pinfo.idx, ...] = 1

            # Convert to TTs
            Intg_int = tntt.TT(
                [pidx]
                + tntt.TT(
                    Intg_int,
                    shape=2 * [(Intg_int.shape[0], Intg_int.shape[1])],
                    eps=self._eps,
                ).cores,
                eps=0,
            )
            self._append_info("Intg_int", Intg_int)

            if "H" in self._only:
                Intg_str.unsqueeze_(-1)
                Intg_str = tntt.TT(
                    [pidx]
                    + tntt.TT(
                        Intg_str,
                        shape=2 * [(Intg_str.shape[0], Intg_str.shape[1])] + [(2, 1)],
                        eps=self._eps,
                    ).cores,
                    eps=0,
                )
                self._append_info("Intg_str", Intg_str)

            return Intg_int, Intg_str

        self._append_info("JRT", JRT)

        # Transpose for application
        R = self._diagonalize(R, [0, 1])
        R.set_core(2, tn.permute(R.cores[2], (0, 2, 1, 3)))
        R.set_core(3, tn.permute(R.cores[3], (0, 2, 1, 3)))
        RJRT = tntt.TT(
            tntt.amen_mm(R, JRT, nswp=50, eps=self._eps).round(self._eps).cores
            + [tn.ones(1).reshape((1, 1, 1, 1))]
        )
        self._append_info("RJRT", RJRT)

        dRJRT = None
        if "H" in self._only:
            # Calculate J @ dR
            JdR = (
                self._diagonalize(J, np.arange(len(J.cores) - 1))
                .fast_matvec(dR, self._eps)
                .round(self._eps)
            )
            JdR = self._diagonalize(JdR, [0, 1, len(JdR.cores) - 1])

            # Transpose quadrature dimensions for sum
            JdR.set_core(2, tn.permute(JdR.cores[2], (0, 2, 1, 3)))
            JdR.set_core(3, tn.permute(JdR.cores[3], (0, 2, 1, 3)))
            self._append_info("JdR", JdR)

            dRJRT = tntt.amen_mm(
                JdR,
                tntt.TT(JRT.cores + [tn.ones(2).reshape((1, 2, 1, 1))]),
                nswp=50,
                eps=self._eps,
            ).round(self._eps)
            self._append_info("dRJRT", dRJRT)

        Intg_int = self._concat_integrals(pinfo, RJRT)
        Intg_str = self._concat_integrals(pinfo, dRJRT) if "H" in self._only else None

        self._append_info("Intg_int", Intg_int)
        if "H" in self._only:
            self._append_info("Intg_str", Intg_str)

        return Intg_int, Intg_str

    def _JRT(self, J_det, R):
        if not self._use_tt:
            return super()._JRT(J_det, R)

        # Calculate (J_det * R^T)
        JRT = tntt.fast_hadammard(J_det, R, self._eps).to_ttm()
        JRT.set_core(4, tn.permute(JRT.cores[4], (0, 2, 1, 3)))
        JRT.set_core(5, tn.permute(JRT.cores[5], (0, 2, 1, 3)))
        JRT = JRT.round(self._eps)
        return JRT

    def _build_loss(self, pinfo: PatchInfo, Intg_int, Intg_str):
        """"""
        # Add angular and energy dimensions
        H = (
            tntt.TT(
                [
                    tn.ones(self._quadrants.shape[0]).reshape((1, -1, 1, 1)),
                    tn.ones(self._ordinates[0].shape[0]).reshape((1, -1, 1, 1)),
                    tn.ones(self._ordinates[1].shape[0]).reshape((1, -1, 1, 1)),
                    tn.tensor(self._xs_server.total(pinfo.patch.material)).reshape(
                        (1, -1, 1, 1)
                    ),
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
                ]
                + Intg_str.cores[:-1]
                + [Intg_str.cores[-1][:, [1], :, :]],
                eps=0,
            )
        )

        # Remove single size dimensions
        H.reduce_dims(exclude=np.arange(len(H.cores) - 1))
        self._append_info("H", H)
        return H

    def _build_scatter(self, pinfo: PatchInfo, Intg_int):
        """"""
        # Check if scattering is zero in this patch
        if self._xs_server.scatter_gtg(pinfo.patch.material) is None:
            return None

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
                        self._xs_server.scatter_gtg(pinfo.patch.material)[
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
        S = tntt.TT(S.cores + Intg_int.cores)
        self._append_info("S", S)
        return S

    def _build_fission(self, pinfo: PatchInfo, Intg_int):
        """"""
        if self._xs_server.nu_fission(pinfo.patch.material) is None:
            return None

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
                    tn.tensor(self._xs_server.nu_fission(pinfo.patch.material)),
                )
                .unsqueeze_(0)
                .unsqueeze_(-1),
            ]
            + Intg_int.cores,
            eps=0,
        ).round(self._eps)
        F.reduce_dims(exclude=np.arange(len(F.cores) - 1))
        self._append_info("F", F)
        return F

    def _build_outgoing_boundary(self, pinfo: PatchInfo, J_det, R):
        """"""
        # Get angular data
        Ox, Oy = self._angular(pinfo, dir=1.0)

        Intg = self._build_boundary_integrals(pinfo, J_det, R, (Ox, Oy), "out")
        self._append_info("Intg_out", Intg)
        del Ox, Oy

        # Append energy identity matrix and permute TT
        B_out = tntt.permute(
            tntt.TT(
                [tn.ones(self._xs_server.num_groups).reshape((1, -1, 1, 1))]
                + Intg.cores,
                eps=0,
            ),
            [2, 3, 4, 0, 1, 5, 6],
            eps=self._eps,
        ).round(self._eps)
        self._append_info("B_out", B_out)
        return B_out

    def _build_incident_boundary(self, pinfo: PatchInfo, J_det, R):
        """"""
        # Unpack
        Jx_det, Jy_det = J_det
        Rx, Ry = R

        # Get angular data
        Ox, Oy = self._angular(pinfo, dir=-1.0)

        if self._use_tt:
            for bidx in range(2):
                # Get adjacent patch index
                xpid = self._mesh.get_connected_patch(pinfo.id, centroid=(0.5, bidx))
                ypid = self._mesh.get_connected_patch(pinfo.id, centroid=(bidx, 0.5))

                # Make mask
                mask = tn.tensor([0, 1] if bidx == 0 else [1, 0]).reshape((1, -1, 1))

                if xpid is None:
                    # Apply mask
                    Jx_det.set_core(0, Jx_det.cores[0].clone() * mask)
                    Rx.set_core(0, Rx.cores[0].clone() * mask)
                    Ox.set_core(3, Ox.cores[3].clone() * mask)

                if ypid is None:
                    # Apply mask
                    Jy_det.set_core(0, Jy_det.cores[0].clone() * mask)
                    Ry.set_core(0, Ry.cores[0].clone() * mask)
                    Oy.set_core(3, Oy.cores[3].clone() * mask)

            # Round results
            Jx_det, Jy_det = Jx_det.round(self._eps), Jy_det.round(self._eps)
            Rx, Ry = Rx.round(self._eps), Ry.round(self._eps)
            Ox, Oy = Ox.round(self._eps), Oy.round(self._eps)

        else:
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
        Intg = self._build_boundary_integrals(
            pinfo, (Jx_det, Jy_det), (Rx, Ry), (Ox, Oy), "in"
        )
        self._append_info("Intg_in", Intg)

        B_in = tntt.permute(
            tntt.TT(
                [tn.eye(self._xs_server.num_groups).unsqueeze_(0).unsqueeze_(-1)]
                + Intg.cores,
                eps=0,
            ),
            [2, 3, 4, 0, 1, 5, 6],
            eps=self._eps,
        ).round(self._eps)
        self._append_info("B_in", B_in)
        return self._diagonalize(B_in, [1, 2])

    def _build_boundary_integrals(self, pinfo: PatchInfo, J_det, R, Or, tag):
        """"""
        # Unpack
        Jx_det, Jy_det = J_det[0].clone(), J_det[1].clone()
        Rx, Ry = R[0].clone(), R[1].clone()
        Ox, Oy = Or

        if not self._use_tt:
            # Calculate local boundary integrals
            local_Intg_x = tn.einsum(
                "kij,j,qnmkij,kijab,kijcd->qnmkiabcd",
                (Jx_det, tn.tensor(self._wx), Ox, Rx, Rx),
            )
            local_Intg_x = tntt.TT(
                local_Intg_x.reshape(
                    (
                        *local_Intg_x.shape[:-2],
                        *((local_Intg_x.ndim - 2) * [1]),
                        *local_Intg_x.shape[-2:],
                    )
                ),
                shape=[(s, 1) for s in local_Intg_x.shape[:-4]]
                + [(s, s) for s in local_Intg_x.shape[-2:]],
                eps=self._eps,
            )
            local_Intg_y = tn.einsum(
                "kij,j,qnmkij,kijab,kijcd->qnmkiabcd",
                (Jy_det, tn.tensor(self._wy), Oy, Ry, Ry),
            )
            local_Intg_y = tntt.TT(
                local_Intg_y.reshape(
                    (
                        *local_Intg_y.shape[:-2],
                        *((local_Intg_y.ndim - 2) * [1]),
                        *local_Intg_y.shape[-2:],
                    )
                ),
                shape=[(s, 1) for s in local_Intg_y.shape[:-4]]
                + [(s, s) for s in local_Intg_y.shape[-2:]],
                eps=self._eps,
            )
            del Jx_det, Jy_det, Ox, Oy, Rx, Ry

        else:
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
                    tn.ones(pinfo.patch.degree[0] + 1).reshape((1, -1, 1)),
                    tn.ones(pinfo.patch.degree[1] + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )
            Oy = tntt.TT(
                Oy.cores
                + [
                    tn.ones(pinfo.patch.degree[0] + 1).reshape((1, -1, 1)),
                    tn.ones(pinfo.patch.degree[1] + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )

            # Hadamard product with angular component
            ORx = tntt.fast_hadammard(Ox, Rx, self._eps)
            ORy = tntt.fast_hadammard(Oy, Ry, self._eps)
            del Ox, Oy
            self._append_info(f"ORx_{tag}", ORx)
            self._append_info(f"ORy_{tag}", ORy)

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
                    tn.ones(pinfo.patch.degree[0] + 1).reshape((1, -1, 1)),
                    tn.ones(pinfo.patch.degree[1] + 1).reshape((1, -1, 1)),
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
                    tn.ones(pinfo.patch.degree[0] + 1).reshape((1, -1, 1)),
                    tn.ones(pinfo.patch.degree[1] + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )

            # Hadamard product with Jacobian
            ORJx_det = tntt.fast_hadammard(ORx, Jx_det, self._eps)
            ORJy_det = tntt.fast_hadammard(ORy, Jy_det, self._eps)
            del Jx_det, Jy_det
            self._append_info(f"ORJx_{tag}", ORJx_det)
            self._append_info(f"ORJy_{tag}", ORJy_det)

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
                pinfo,
                [0, 1],
                local_Intg_x,
                connected_patches=(
                    [
                        self._mesh.get_connected_patch(pinfo.id, (0.5, i))
                        for i in range(2)
                    ]
                    if tag == "in"
                    else None
                ),
            )
            + self._concat_boundary_integrals(
                pinfo,
                [2, 3],
                local_Intg_y,
                connected_patches=(
                    [
                        self._mesh.get_connected_patch(pinfo.id, (i, 0.5))
                        for i in range(2)
                    ]
                    if tag == "in"
                    else None
                ),
            )
        ).round(self._eps)
        Intg.reduce_dims(exclude=[0, 1, 2, 3, 7, 8])
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
    def _jacobian(self, pinfo: PatchInfo):
        """"""
        if not self._use_tt:
            return super()._jacobian(pinfo)

        if self.interp_jacobian:
            raise NotImplementedError()
            # # Interpolate Jacobian matrix at each quadrature point
            # def _interp_jacobian(i, j):
            #     indicator = tn.zeros((1, 2, 2, 1))
            #     indicator[:, i, j, :] = 1
            #     return tntt.TT(
            #         tntt.interpolate.dmrg_cross(
            #             lambda idxs: self._sample_jacobian(idxs)[i, j, :],
            #             [self._I1, self._I2, *self._num_points],
            #             eps=self._eps,
            #             nswp=50,
            #         )
            #         .to_ttm()
            #         .cores
            #         + [
            #             tn.ones(self._patch.degree_u + 1).reshape((1, -1, 1, 1)),
            #             tn.ones(self._patch.degree_v + 1).reshape((1, -1, 1, 1)),
            #             indicator,
            #         ],
            #         eps=0,
            #     )
            #
            # i, j = np.meshgrid(np.arange(2), np.arange(2))
            # return sum(
            #     np.vectorize(_interp_jacobian, otypes=[tntt.TT])(i, j).flatten()
            # ).round(self._eps)

        else:
            # Create TT for each subelement
            i1, i2 = np.meshgrid(np.arange(pinfo.I1), np.arange(pinfo.I2))

            # Sum all subelement TTs and round
            J = sum(
                map(
                    self._calc_jacobian,
                    itertools.repeat(pinfo),
                    i1.flatten(),
                    i2.flatten(),
                )
            ).round(self._eps)

            # Add cores for non-vanishing basis functions and permute Jacobian
            # matrix to the end
            J = tntt.permute(
                tntt.TT(
                    J.cores
                    + [
                        tn.ones(pinfo.patch.degree[0] + 1).reshape((1, -1, 1, 1)),
                        tn.ones(pinfo.patch.degree[1] + 1).reshape((1, -1, 1, 1)),
                    ],
                    eps=0,
                ),
                dims=list(range(len(J.cores) - 1))
                + [len(J.cores) + i - 1 for i in range(1, 3)]
                + [len(J.cores) - 1],
                eps=self._eps,
            )

        self._append_info("J", J)
        return J

    def _jacobian_det(self, pinfo: PatchInfo):
        """"""
        if not self._use_tt:
            return super()._jacobian_det(pinfo)

        if self.interp_jacobian_det:
            raise NotImplementedError()
            # # Apply cross approximation
            # return tntt.interpolate.dmrg_cross(
            #     self._sample_jacobian_det,
            #     [self._I1, self._I2, *self._num_points],
            #     eps=self._eps,
            #     nswp=50,
            # ).round(self._eps)

        elif self._use_tt:
            # Create TT for each subelement
            i1, i2 = np.meshgrid(np.arange(pinfo.I1), np.arange(pinfo.I2))

            # Vectorize evaluation function and run
            J_det = sum(
                map(
                    self._calc_jacobian_det,
                    itertools.repeat(pinfo),
                    i1.flatten(),
                    i2.flatten(),
                )
            ).round(self._eps)

            J_det = tntt.TT(
                J_det.cores
                + [
                    tn.ones(pinfo.patch.degree[0] + 1).reshape((1, -1, 1)),
                    tn.ones(pinfo.patch.degree[1] + 1).reshape((1, -1, 1)),
                ],
                eps=0,
            )

            # Hadamard product with the weights
            J_det.cores[2] *= tn.tensor(self._wx).reshape((1, -1, 1))
            J_det.cores[3] *= tn.tensor(self._wy).reshape((1, -1, 1))
            J_det = J_det.round(self._eps)

        self._append_info("J_det", J_det)
        return J_det

    def _basis(self, pinfo: PatchInfo):
        """"""
        if not self._use_tt:
            return super()._basis(pinfo)

        if self.interp_basis:
            raise NotImplementedError()

        elif self._use_tt:
            # Evaluate basis and their derivatives at each subelement
            i1, i2 = np.meshgrid(np.arange(pinfo.I1), np.arange(pinfo.I2))
            result = list(
                map(
                    self._calc_basis,
                    itertools.repeat(pinfo),
                    i1.flatten(),
                    i2.flatten(),
                )
            )
            R, dR = sum(x[0] for x in result).round(self._eps), sum(
                x[1] for x in result
            ).round(self._eps)

        self._append_info("R", R)
        self._append_info("dR", dR)
        return R, dR

    def _boundary_jacobian_det(self, pinfo: PatchInfo):
        """"""
        if not self._use_tt:
            return super()._boundary_jacobian_det(pinfo)

        if self.interp_boundary_jacobian_det:
            raise NotImplementedError()
            # # Apply cross approximation and sum boundaries
            # boundary_jacobian_func = np.vectorize(
            #     lambda i: tntt.TT(
            #         [
            #             tn.tensor([1, 0] if (i % 2) == 0 else [0, 1], dtype=tn.float64)
            #             .unsqueeze_(0)
            #             .unsqueeze_(-1)
            #         ]
            #         + tntt.interpolate.dmrg_cross(
            #             lambda idxs: self._sample_boundary_jacobian_det(i, idxs),
            #             (
            #                 [self._I1, self._num_points[0]]
            #                 if i < 2
            #                 else [self._I2, self._num_points[1]]
            #             ),
            #             eps=self._eps,
            #             nswp=50,
            #         ).cores,
            #         eps=0,
            #     ),
            #     otypes=[tntt.TT],
            # )
            #
            # return sum(boundary_jacobian_func([0, 1])).round(self._eps), sum(
            #     boundary_jacobian_func([2, 3])
            # ).round(self._eps)

        else:
            # Evaluate boundary Jacobian
            xboundary_idxs, i1 = np.meshgrid(np.arange(2), np.arange(pinfo.I1))
            yboundary_idxs, i2 = np.meshgrid(np.arange(2, 4), np.arange(pinfo.I2))

            Jx_det, Jy_det = sum(
                map(
                    self._calc_boundary_jacobian_det,
                    itertools.repeat(pinfo),
                    xboundary_idxs.flatten(),
                    i1.flatten(),
                )
            ).round(self._eps), sum(
                map(
                    self._calc_boundary_jacobian_det,
                    itertools.repeat(pinfo),
                    yboundary_idxs.flatten(),
                    i2.flatten(),
                )
            ).round(
                self._eps
            )

        self._append_info("Jx_det", Jx_det)
        self._append_info("Jy_det", Jy_det)
        return Jx_det, Jy_det

    def _boundary_basis(self, pinfo: PatchInfo):
        """"""
        if not self._use_tt:
            return super()._boundary_basis(pinfo)

        if self.interp_boundary_basis:
            raise NotImplementedError()

        else:
            # Evaluate boundary Jacobian
            xbidxs, i1 = np.meshgrid(np.arange(2), np.arange(pinfo.I1))
            ybidxs, i2 = np.meshgrid(np.arange(2, 4), np.arange(pinfo.I2))

            Rx, Ry = sum(
                map(
                    self._calc_boundary_basis,
                    itertools.repeat(pinfo),
                    xbidxs.flatten(),
                    i1.flatten(),
                )
            ).round(self._eps), sum(
                map(
                    self._calc_boundary_basis,
                    itertools.repeat(pinfo),
                    ybidxs.flatten(),
                    i2.flatten(),
                )
            ).round(
                self._eps
            )

        self._append_info("Rx", Rx)
        self._append_info("Ry", Ry)
        return Rx, Ry

    def _angular(self, pinfo: PatchInfo, dir):
        """"""
        if not self._use_tt:
            return super()._angular(pinfo, dir)

        if self.interp_angular:
            raise NotImplementedError()

        else:
            # Evaluate boundary Jacobian
            xboundary_idxs, i1 = np.meshgrid(np.arange(2), np.arange(pinfo.I1))
            yboundary_idxs, i2 = np.meshgrid(np.arange(2, 4), np.arange(pinfo.I2))

            # Sum over boundary subelements
            Ox = sum(
                map(
                    self._calc_angular,
                    itertools.repeat(pinfo),
                    xboundary_idxs.flatten(),
                    itertools.repeat(dir),
                    i1.flatten(),
                )
            ).round(self._eps)
            Oy = sum(
                map(
                    self._calc_angular,
                    itertools.repeat(pinfo),
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

        self._append_info("Ox", Ox)
        self._append_info("Oy", Oy)
        return Ox, Oy

    # ========================================================================
    # Non-interpolation methods

    def _calc_jacobian(self, pinfo: PatchInfo, i1, i2):
        """"""
        J = super()._calc_jacobian(pinfo, i1, i2)
        if not self._use_tt:
            return J

        # Convert to TT format
        J = J.reshape((*self._num_points, 2, 1, 1, 2))
        J = tntt.TT(
            J, shape=[(J.shape[i], 1) for i in range(2)] + [(2, 2)], eps=self._eps
        )

        # Apply Kronecker product
        indx = tn.zeros(pinfo.I1).reshape((1, -1, 1, 1))
        indy = tn.zeros(pinfo.I2).reshape((1, -1, 1, 1))
        indx[0, i1, 0, 0] = 1
        indy[0, i2, 0, 0] = 1

        return tntt.TT([indx, indy] + J.cores, eps=0)

    def _calc_jacobian_det(self, pinfo: PatchInfo, i1, i2):
        """"""
        J_det = super()._calc_jacobian_det(pinfo, i1, i2)
        if not self._use_tt:
            return J_det

        # Convert to TT format
        J_det = tntt.TT(J_det, eps=self._eps)

        # Apply Kronecker product
        indx = tn.zeros(pinfo.I1).reshape((1, -1, 1))
        indy = tn.zeros(pinfo.I2).reshape((1, -1, 1))
        indx[0, i1, 0] = 1
        indy[0, i2, 0] = 1

        return tntt.TT([indx, indy] + J_det.cores, eps=0)

    def _calc_basis(self, pinfo: PatchInfo, i1, i2):
        """"""
        R, dR = super()._calc_basis(pinfo, i1, i2)
        if not self._use_tt:
            return R, dR

        # Convert to TT format
        R = tntt.TT(R, eps=self._eps)
        dR = tntt.TT(dR, eps=self._eps)

        # Apply Kronecker product
        indx = tn.zeros(pinfo.I1).reshape((1, -1, 1))
        indy = tn.zeros(pinfo.I2).reshape((1, -1, 1))
        indx[0, i1, 0] = 1
        indy[0, i2, 0] = 1

        return tntt.TT([indx, indy] + R.cores, eps=0), tntt.TT(
            [indx, indy] + dR.cores, eps=0
        )

    def _calc_boundary_jacobian_det(self, pinfo: PatchInfo, bidx, i):
        """"""
        J_det = super()._calc_boundary_jacobian_det(pinfo, bidx, i)
        if not self._use_tt:
            return J_det

        # Convert to TT format
        indbc = tn.zeros(2).reshape((1, -1, 1))
        indxy = tn.zeros(pinfo.I1 if bidx < 2 else pinfo.I2).reshape((1, -1, 1))
        indbc[0, bidx % 2, 0] = 1
        indxy[0, i, 0] = 1

        return tntt.TT([indbc, indxy, J_det.reshape(1, -1, 1)], eps=0)

    def _calc_boundary_basis(self, pinfo: PatchInfo, bidx, i):
        """"""
        R = super()._calc_boundary_basis(pinfo, bidx, i)
        if not self._use_tt:
            return R

        # Create indicator
        indicator = tn.zeros(pinfo.I1 if bidx < 2 else pinfo.I2)
        indicator[i] = 1.0

        return tntt.TT(
            [
                tn.tensor(
                    [1, 0] if bidx % 2 == 0 else [0, 1], dtype=tn.float64
                ).reshape((1, -1, 1)),
                indicator.reshape((1, -1, 1)),
            ]
            + tntt.TT(R, eps=self._eps).cores,
            eps=0,
        )

    def _calc_angular(self, pinfo, bidx, dir, i):
        """"""
        products = super()._calc_angular(pinfo, bidx, dir, i)
        if not self._use_tt:
            return products

        # Create indicator
        ind = tn.zeros(pinfo.I1 if bidx < 2 else pinfo.I2)
        ind[i] = 1.0

        return tntt.TT(
            tntt.TT(tn.sqrt(products), eps=self._eps).cores
            + [
                tn.tensor(
                    [1, 0] if bidx % 2 == 0 else [0, 1], dtype=tn.float64
                ).reshape((1, -1, 1)),
                ind.reshape((1, -1, 1)),
            ]
        )

    def _concat_integrals(self, pinfo: PatchInfo, local_Intg):
        """"""
        # Create TT for each subelement
        i1, i2 = np.meshgrid(np.arange(pinfo.I1), np.arange(pinfo.I2))

        # Sum and round
        Intg = sum(
            map(
                self._get_local_Intg,
                itertools.repeat(pinfo),
                i1.flatten(),
                i2.flatten(),
                itertools.repeat(local_Intg),
            )
        ).round(self._eps)
        Intg.reduce_dims()

        # Add patch index
        pidx = tn.zeros(self._mesh.num_patches).reshape((1, -1, 1, 1))
        pidx[0, pinfo.idx, ...] = 1
        return tntt.TT([pidx] + Intg.cores, eps=0)

    def _concat_boundary_integrals(
        self, pinfo: PatchInfo, bidxs, local_Intg_bound, connected_patches=None
    ):
        """"""
        # Get integrals over each boundary
        Intg = [
            sum(
                map(
                    self._get_local_boundary_Intg,
                    itertools.repeat(pinfo),
                    itertools.repeat(bidx),
                    np.arange(pinfo.I1 if bidx < 2 else pinfo.I2),
                    itertools.repeat(local_Intg_bound),
                )
            ).round(self._eps)
            for bidx in bidxs
        ]

        # Handle incident boundary conditions
        if connected_patches is not None:
            for bidx, pid in zip(bidxs, connected_patches):
                # Patch indicator
                pidx = tn.zeros((1, self._mesh.num_patches, self._mesh.num_patches, 1))

                if pid == pinfo.id:
                    pidx[0, pinfo.idx, pinfo.idx, 0] = 1

                    # Handle reflective boundary condition
                    # Initialize core
                    quadrant_core = tn.zeros((1, 4, 4, Intg[bidx % 2].R[1]))

                    # Iterate through quadrants
                    for quad in range(4):
                        quadrant = self._quadrants[quad, :].copy()

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
                        ref_quad = (
                            (self._quadrants == tuple(quadrant))
                            .all(axis=1)
                            .nonzero()[0][0]
                        )

                        # Place quadrant
                        quadrant_core[0, quad, ref_quad, :] = Intg[bidx % 2].cores[0][
                            0, quad, 0, :
                        ]

                    # Replace quadrant core
                    Intg[bidx % 2].set_core(0, quadrant_core)

                else:
                    if pid is not None:
                        # Indicate connected patch
                        pidx[0, pinfo.idx, self._mesh.pid2pidx(pid), 0] = 1

                        # Reflect across opposite axis core
                        Intg[bidx % 2].set_core(
                            len(Intg[bidx % 2].cores) - (1 if bidx < 2 else 2),
                            tn.flip(
                                Intg[bidx % 2].cores[-1 if bidx < 2 else -2],
                                dims=[2],
                            ),
                        )

                    # Handle vacuum boundary condition
                    Intg[bidx % 2] = self._diagonalize(Intg[bidx % 2], [0])

                # Append patch core
                Intg[bidx % 2] = tntt.TT([pidx] + Intg[bidx % 2].cores, eps=0)

            return sum(Intg).round(self._eps)

        else:
            # Get patch core
            pidx = tn.zeros(self._mesh.num_patches).reshape((1, -1, 1, 1))
            pidx[0, pinfo.idx, ...] = 1

            return tntt.TT([pidx] + sum(Intg).round(self._eps).cores, eps=0)

    def _get_local_Intg(self, pinfo: PatchInfo, i1, i2, Intg):
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
                pinfo.patch.shape[0],
                pinfo.patch.shape[0],
                Intg.R[-3],
            )
        )
        yhat = tn.zeros(
            (
                Intg.R[-3],
                pinfo.patch.shape[1],
                pinfo.patch.shape[1],
                Intg.R[-2],
            )
        )

        # Place smaller core
        xhat[
            :,
            i1 : i1 + pinfo.patch.degree[0] + 1,
            i1 : i1 + pinfo.patch.degree[0] + 1,
            :,
        ] = Intg.cores[-3]
        yhat[
            :,
            i2 : i2 + pinfo.patch.degree[1] + 1,
            i2 : i2 + pinfo.patch.degree[1] + 1,
            :,
        ] = Intg.cores[-2]

        return tntt.TT(cores + [xhat, yhat, Intg.cores[-1]], eps=0).round(self._eps)

    def _get_local_boundary_Intg(self, pinfo: PatchInfo, bidx, i, Intg_bound):
        """"""
        # Index out local integral
        cores = [
            *Intg_bound.cores[:3],
            Intg_bound.cores[3][:, [bidx % 2], :, :],
            Intg_bound.cores[4][:, [i], :, :],
            *Intg_bound.cores[5:-2],
        ]

        # Increase spatial core size to all control points
        xhat = tn.zeros(
            (
                Intg_bound.R[-3],
                pinfo.patch.shape[0],
                pinfo.patch.shape[0],
                Intg_bound.R[-2],
            )
        )
        yhat = tn.zeros(
            (
                Intg_bound.R[-2],
                pinfo.patch.shape[1],
                pinfo.patch.shape[1],
                Intg_bound.R[-1],
            )
        )

        # Place smaller core
        if bidx < 2:
            i1 = i
            i2 = 0 if bidx == 0 else pinfo.I2 - 1
        else:
            i1 = 0 if bidx == 2 else pinfo.I1 - 1
            i2 = i

        xhat[
            :,
            i1 : i1 + pinfo.patch.degree[0] + 1,
            i1 : i1 + pinfo.patch.degree[0] + 1,
            :,
        ] = Intg_bound.cores[-2]
        yhat[
            :,
            i2 : i2 + pinfo.patch.degree[1] + 1,
            i2 : i2 + pinfo.patch.degree[1] + 1,
            :,
        ] = Intg_bound.cores[-1]

        return tntt.TT(cores + [xhat, yhat], eps=0).round(self._eps)

    # ========================================================================
    # I/O

    def _print_patch(self, pid):
        """"""
        if self._verbose:
            print(f"Assembling Patch {pid}")
            print(
                "{:15s} {:25s} {:10s}  {:15s}".format(
                    "Step", "Ranks", "Compression", "Elapsed Time (s)"
                )
            )

    def _print_info(self, name, data, final):
        """"""
        if not final:
            print(
                "{:15s} {:25s} {:10.2f}  {:10.2f}".format(
                    name,
                    ",".join(map(str, data["ranks"])),
                    data["compression"],
                    data["elapsed time"],
                )
            )
        else:
            print(
                "{:15s} {:25s} {:10.2f}".format(
                    name,
                    ",".join(map(str, data["ranks"])),
                    data["compression"],
                )
            )

    # ========================================================================
    # Other

    def _append_info(self, name, A, final=False):
        """"""
        # Get info
        info = (
            {
                "shape": A.shape,
                "ranks": A.R[1:-1],
                "entries": sum([tn.numel(c) for c in A.cores]),
                "compression": self.compression(A),
                "elapsed time": time.time() - self._start_time,
            }
            if isinstance(A, tntt.TT)
            else {
                "shape": A.shape,
                "ranks": [np.nan],
                "entries": np.prod(A.shape),
                "compression": 1,
                "elapsed time": time.time() - self._start_time,
            }
        )
        if final:
            self._assembly_info[name] = info

        # Print info
        if self._verbose and (final or self._max_processes == 1):
            self._print_info(name, info, final)

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
