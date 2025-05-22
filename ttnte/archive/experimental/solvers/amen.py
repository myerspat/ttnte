import itertools
import warnings
from typing import Optional, Union

import cupy as cu
import numpy as np
from autoray import do
from cupy.linalg import svd as cu_svd
from cupyx.scipy.sparse.linalg import gmres as cu_gmres
from quimb.core import prod
from quimb.tensor import (
    MatrixProductOperator,
    MatrixProductState,
    MPS_rand_state,
    Tensor,
)
from quimb.tensor.tensor_core import Tensor
from scipy.linalg import qr
from scipy.linalg import svd as sc_svd
from scipy.sparse.linalg import gmres as sc_gmres

from ttnte.archive.experimental.solvers._dmrg import MovingEnvironment
from ttnte.archive.experimental.solvers.gpu_manager import GPUManager
from ttnte.archive.experimental.solvers.linear_operator import TNLinearOperator

# TODO: This implementation only supports the ALS kick type,
# add others later


class AMEn(object):
    def __init__(
        self,
        verbose: bool = False,
        **opts,
    ):
        """"""
        self._verbose = verbose
        self._set_opts(opts)

    def _set_opts(self, opts):
        """"""
        # Base solver options
        self._use_gpu = opts.get("use_gpu", False)
        self._local_prec = opts.get("local_prec", "")
        self._local_iters = opts.get("local_iters", 2)
        self._local_restart = opts.get("local_restart", 40)
        self._res_damp = opts.get("res_damp", 2)
        self._trunc_norm = opts.get("trunc_norm", "residual")
        self._tol_exit = opts.get("tol_exit", "tol")
        self._max_full_size = opts.get("max_full_size", 2500)
        self._min_system_size = opts.get("min_system_size", 2500)

        # Create GPU memory manager
        self._gpu_manager = (
            GPUManager(
                safety_factor=opts.get("safety_factor", 0.9),
                device_id=opts.get("device_id", 0),
            )
            if opts.get("use_gpu", False)
            else None
        )

    def _setup(
        self,
        A: MatrixProductOperator,
        B: MatrixProductState,
        x0: Optional[MatrixProductState] = None,
        z0: Optional[MatrixProductState] = None,
        tol: float = 1e-5,
        kickrank: int = 4,
        kicktype: str = "als",
        kickrank2: int = 0,
        virtual: bool = True,
    ):
        """"""
        # Save params
        self._tol = tol
        self._kicktype = kicktype
        self._kickrank = kickrank
        self._kickrank2 = kickrank2
        self._sweep_num = 0
        self._converged = False

        # Get length of network
        self._order = A.L
        self._phys_dims = [A.phys_dim(site=site) for site in A.sites]

        # Tighter tolerance to stop local problems with noise
        self._real_tol = self._tol / np.sqrt(self._order) / self._res_damp

        # Check given TNs
        self._check_tns(B, x0, z0)

        # Save all operators and vectors
        self._A = A.copy(virtual=virtual)
        self._B = B.copy(virtual=virtual)
        self._x = (
            x0.copy()
            if x0
            else MPS_rand_state(
                L=self._order,
                bond_dim=2,
                phys_dim=self._phys_dims,
                dist="uniform",
                normalize="left",
            )
        )
        z = (
            z0.copy()
            if z0
            else MPS_rand_state(
                L=self._order,
                bond_dim=self._kickrank + self._kickrank2,
                phys_dim=self._phys_dims,
                dist="uniform",
                normalize="left",
            )
        )

        # Ensure correct rank
        # self._x.compress(max_bond=2)
        # z.compress(max_bond=self._kickrank + self._kickrank2)

        # Permuate arrays to lpr and ludr
        self._A.permute_arrays("ludr")
        self._B.permute_arrays("lpr")
        self._x.permute_arrays("lpr")
        z.permute_arrays("lpr")

        # Equalize Norms
        self._A.equalize_norms(inplace=True)
        self._B.equalize_norms(inplace=True)
        self._x.equalize_norms(inplace=True)
        z.equalize_norms(inplace=True)

        # Save original physical index tags
        self._inds = [self._x[0].inds[0]] + [core.inds[1] for core in self._x[1:]]

        # Create conjugate transposes
        self._xH = self._x.H
        self._zH = z.H

        # Left and right orthogonalize z and x
        self._x.left_canonize(bra=self._xH, inplace=True)
        z.left_canonize(bra=self._zH, inplace=True)
        self._x.right_canonize(bra=self._xH, inplace=True)
        z.right_canonize(bra=self._zH, inplace=True)

        # Tag all TNs
        self._A.add_tag("_A")
        self._B.add_tag("_B")
        self._x.add_tag("_x")
        self._xH.add_tag("_xH")
        self._zH.add_tag("_zH")

        # Align systems
        self._xH.align_(self._A, self._x)
        self._xH.align_(self._B)
        self._zH.align_(self._A, self._x)
        self._zH.align_(self._B)

        # Create projections
        self._LHS = self._xH | self._A | self._x
        self._RHS = self._xH | self._B
        self._zLHS = self._zH | self._A | self._x
        self._zRHS = self._zH | self._B

        # Maximums
        self._max_dx = 0
        self._max_res = 0

    def _check_tns(self, B, x0, z0):
        assert self._order
        assert self._phys_dims
        assert B

        # Check shapes of given data
        check = self._order == B.L
        check = self._phys_dims == [B.phys_dim(site=site) for site in B.sites]
        if x0:
            check = self._order == x0.L
            check = self._phys_dims == [x0.phys_dim(site=site) for site in x0.sites]
        if z0:
            check = self._order == z0.L
            check = self._phys_dims == [z0.phys_dim(site=site) for site in z0.sites]

        if not check:
            raise RuntimeError(
                "A, B, x0, and z0 must have the same order and physical dimensions"
            )

    # =============================================================
    # Base solve method

    def solve(
        self,
        A: MatrixProductOperator,
        B: MatrixProductState,
        x0: Optional[MatrixProductState] = None,
        z0: Optional[MatrixProductState] = None,
        tol: float = 1e-5,
        max_ranks=1000,
        max_sweeps: int = 50,
        kickrank: int = 4,
        kicktype: str = "als",
        kickrank2: int = 0,
        virtual: bool = True,
        verbose: Optional[bool] = None,
        **opts,
    ):
        """"""
        self._verbose = verbose if verbose else self._verbose
        if opts:
            self._set_opts(opts)

        # Determine initial rank
        self._max_rank, max_ranks = self._set_seq(max_ranks)

        # Setup AMEn solver
        self._setup(
            A=A,
            B=B,
            x0=x0,
            z0=z0,
            tol=tol,
            kickrank=kickrank,
            kicktype=kicktype,
            kickrank2=kickrank2,
            virtual=virtual,
        )

        # Begin AMEn sweeps
        i = 0
        for i in range(max_sweeps - 1):
            # Get next rank
            self._max_rank = next(max_ranks)

            # Print pre-sweep
            self._print_pre_sweep(i)

            # Run sweep
            self._sweep()

            # Print post sweep
            self._print_post_sweep()

            # Check convergence
            if self._max_res < self._tol:
                self._converged = True
                break

        # Run final sweep
        self._print_pre_sweep(i + 1)
        self._sweep()
        self._print_post_sweep()

        # Equalize norms in solution
        self._x.equalize_norms(inplace=True)

        return self.x, self._converged

    # =============================================================
    # Sweep methods

    def _sweep(self):
        """"""
        # Orthogonalize x
        self._x.right_canonize(bra=self._xH, inplace=True)
        # self._right_canonize_x()

        if not self._converged and self._kickrank + self._kickrank2 > 0:
            if self._sweep_num > 0:
                # Create left moving environment
                self._zLHS_ME = MovingEnvironment(
                    tn=self._zLHS, begin="right", num_sites=1
                )
                self._zRHS_ME = MovingEnvironment(
                    tn=self._zRHS, begin="right", num_sites=1
                )

            # Orthogonalize z (use projections to update system)
            self._right_canonize_z()

            # Create right moving environment
            self._zLHS_ME = MovingEnvironment(tn=self._zLHS, begin="left", num_sites=1)
            self._zRHS_ME = MovingEnvironment(tn=self._zRHS, begin="left", num_sites=1)

        # Increment sweep number
        self._sweep_num += 1

        # Setup local problems
        self._LHS_ME = MovingEnvironment(tn=self._LHS, begin="left", num_sites=1)
        self._RHS_ME = MovingEnvironment(tn=self._RHS, begin="left", num_sites=1)

        # Reset max residual and dx
        self._max_res = 0
        self._max_dx = 0

        # Run local problems
        for i in range(self._order):
            self._update_local_x(i)

    def _update_local_x(self, i):
        """"""
        xeff_prev = do("reshape", self._x[i].to_dense(self._x[i].inds), (-1, 1))

        # Solve local linear system
        Aeff, Beff, xeff, xeff_prev, use_gpu = self._solve_local_system(i, xeff_prev)

        # Compute residual information
        self._compute_residuals(Aeff, Beff, xeff, xeff_prev)

        # Apply xeff truncation
        u, v, xeff = self._truncate_x(i, Aeff, Beff, xeff, use_gpu)

        if use_gpu:
            # Remove everything from GPU
            Beff, xeff, xeff_prev = self._gpu_manager.from_gpu((Beff, xeff, xeff_prev))

            if i < self._order - 1:
                u, v = self._gpu_manager.from_gpu((u, v))

            if isinstance(Aeff, TNLinearOperator):
                self._gpu_manager.from_gpu(self._LHS_ME()["_A"])
            else:
                Aeff = self._gpu_manager.from_gpu(Aeff)

        if not self._converged and self._kickrank + self._kickrank2 > 0:
            # Calculate residual, truncate, and enrich residual (if kickrank2 > 0)
            self._left_canonize_z_site(i, xeff)

        # Enrich core in x
        self._enrich(i, xeff, u, v)

    def _compute_residuals(self, Aeff, Beff, xeff, xeff_prev):
        # Compute norm of RHS, previous residual, new residual, norm of change,
        # max change, and max residual
        self._res_prev = do("linalg.norm", Aeff @ xeff_prev - Beff) / self._norm_rhs
        self._res_new = do("linalg.norm", Aeff @ xeff - Beff) / self._norm_rhs
        self._dx = do("linalg.norm", xeff - xeff_prev) / do("linalg.norm", xeff)
        self._max_dx = self._max_dx if self._max_dx > self._dx else self._dx
        self._max_res = (
            self._max_res if self._max_res > self._res_prev else self._res_prev
        )

        if (
            self._res_new != 0
            and self._res_prev / self._res_new < self._res_damp
            and self._res_new > self._real_tol
        ):
            # If true we may introduce an error larger than the improvement of
            # the local solution, add preconditioner to combat this
            warnings.warn("Residual damp was smaller than in the truncation")

    def _truncate_x(self, i, Aeff, Beff, xeff, use_gpu):
        if (self._kickrank >= 0) and i < self._order - 1:
            # Left matricize xeff
            xeff = Tensor(
                do("reshape", xeff, self._x[i].shape),
                inds=self._x[i].inds,
            )
            xeff, shape, lix, uix = self._left_matricize(i, xeff, self._x)

            # Run SVD if we're truncating
            u, s, v = self._svd(xeff, use_gpu)
            # u, s, v = do("linalg.svd", xeff)

            # Determine rank of acceptable compression
            r = 0
            for r in reversed(range(min(xeff.shape[0], xeff.shape[1]) + 1)):
                # Compute new xeff
                xeff = u[:, :r] @ np.diag(s[:r]) @ v[:r, :]

                # Reshape to match quimb
                xeff = self._left_tensorize(i, xeff, self._x, shape, lix, uix)
                xeff = do("reshape", xeff.to_dense(self._x[i].inds), (-1, 1))

                # Determine residual
                res = do("linalg.norm", Aeff @ xeff - Beff) / self._norm_rhs

                if res > max(self._real_tol * self._res_damp, self._res_new):
                    break

            # Correct rank
            r += 1
            r = min([r, s.size, self._max_rank])

            # Create low rank approximation
            u = u[:, :r]
            v = do("transpose", do("conjugate", v))[:, :r] @ np.diag(s[:r])

            # Reshape to match quimb
            xeff = do("matmul", u, do("transpose", v))
            xeff = self._left_tensorize(i, xeff, self._x, shape, lix, uix)
            xeff = do("reshape", xeff.to_dense(self._x[i].inds), (-1, 1))

            return u[:, :r], v, xeff

        return None, None, xeff

    def _solve_local_system(self, i, xeff_prev):
        # Move environments
        self._LHS_ME.move_to(i)
        self._RHS_ME.move_to(i)

        # Get index pointers for projections
        inds = {
            "ldims": self._xH[i].shape,
            "lix": self._xH[i].inds,
            "rdims": self._x[i].shape,
            "uix": self._x[i].inds,
        }

        # Form local operators
        Aeff, Beff, dense, use_gpu = self._form_local_ops(
            self._LHS_ME(), self._RHS_ME(), inds
        )
        if use_gpu:
            xeff_prev = self._gpu_manager.to_gpu(xeff_prev)

        # Compute norm of RHS
        self._norm_rhs = do("linalg.norm", Beff)

        # Solve system
        if dense:
            xeff = do("linalg.solve", Aeff, Beff)
        else:
            if self._norm_rhs > 0:
                res = Beff - Aeff.matvec(xeff_prev)
                res_norm = do("linalg.norm", res)

                # Calculate gmres relative tolerance
                real_tol = self._real_tol * self._norm_rhs / res_norm

                if real_tol < 1:
                    # TODO: Add preconditioning
                    xeff, _ = self._gmres(
                        A=Aeff,
                        b=res,
                        rtol=real_tol,
                        restart=self._local_restart,
                        maxiter=self._local_iters,
                        use_gpu=use_gpu,
                    )
                    xeff = xeff_prev + do("reshape", xeff, (-1, 1))
                else:
                    xeff = xeff_prev

            else:
                xeff = do("zeros", do("shape", xeff_prev))

        return Aeff, Beff, xeff, xeff_prev, use_gpu

    def _enrich(self, i, xeff, u, v):
        if i < self._order - 1:
            _, shape, lix, uix = self._left_matricize(i, self._x[i], self._x)

            if self._kickrank > 0 and not self._converged:
                # Get indices and TN
                inds = {
                    "left_inds": list(self._zH[i].inds),
                    "right_inds": list(self._x[i].inds),
                    "ldims": list(self._zH[i].shape),
                    "rdims": list(self._x[i].shape),
                }
                tn = self._A[i] | self._zLHS_ME()["_RIGHT"]
                if i > 0:
                    inds["left_inds"][0] = self._xH.bond(i - 1, i)
                    inds["ldims"][0] = self._LHS_ME()["_LEFT"].ind_size(
                        inds["left_inds"][0]
                    )

                    tn = self._LHS_ME()["_LEFT"] | tn

                # Correct shape to old rank
                inds["ldims"][-1] = self._zLHS_ME()["_RIGHT"].ind_size(
                    inds["left_inds"][-1]
                )

                # Compute LHS
                LHS = TNLinearOperator(tns=tn, **inds) @ xeff

                # Setup right side
                RHS = do(
                    "reshape",
                    (
                        self._RHS_ME()["_LEFT"] @ self._B[i] @ self._zRHS_ME()["_RIGHT"]
                    ).to_dense(inds["left_inds"]),
                    (-1, 1),
                )

                # Compute enrichment
                uk = Tensor(
                    do("reshape", RHS - LHS, inds["ldims"]), inds=self._x[i].inds
                )
                uk, shape, lix, uix = self._left_matricize(i, uk, self._x)

                # Apply enrichment
                # u, rv = do("linalg.qr", do("concatenate", [u, uk], axis=-1))
                u, rv = qr(do("concatenate", [u, uk], axis=-1), mode="economic")
                v = do(
                    "concatenate",
                    [v, do("zeros", (self._x.bond_size(i, i + 1), uk.shape[-1]))],
                    axis=-1,
                )
                v = do("matmul", v, do("transpose", rv))

            # Tensorize u
            shape[-1] = u.shape[-1]
            u = self._left_tensorize(i, u, self._x, shape, lix, uix)

            # Multiply by next core
            xr, shape, lix, uix = self._right_matricize(i + 1, self._x[i + 1], self._x)
            v = do("matmul", do("transpose", v), xr)

            # Tensorize v
            shape[0] = v.shape[0]
            v = self._right_tensorize(i + 1, v, self._x, shape, lix, uix)

            # Put back into solution TNs
            self._x[i].modify(data=u.data)
            self._x[i + 1].modify(data=v.data)
            self._xH[i].modify(data=u.conj().data)
            self._xH[i + 1].modify(data=v.conj().data)

        else:
            # For i = self._order - 1 place solution back
            xeff = do("reshape", xeff, self._x[i].shape)
            self._x[i].modify(data=xeff)
            self._xH[i].modify(data=xeff)

    def _form_local_ops(self, LHS_eff, RHS_eff, inds, dense=None):
        """"""
        # Use dense solve if system is less than 800x800
        n = prod(inds["ldims"])
        m = prod(inds["rdims"])
        if n * m < self._max_full_size:
            # Check if system will fit on GPU
            use_gpu = (
                self._gpu_manager.check_available(
                    num_elements=[n * m, n, n],
                    types=[LHS_eff.dtype, RHS_eff.dtype, self._x.dtype],
                )
                if self._gpu_manager
                else False
            )

            if use_gpu:
                self._gpu_manager.to_gpu((LHS_eff["_A"], RHS_eff["_B"]))

            # Compute RHS vector
            RHS_eff_dense = do(
                "reshape", (RHS_eff ^ "_B")["_B"].to_dense(inds["lix"]), (-1, 1)
            )

            # Contract to form matrix operator
            LHS_eff_dense = (LHS_eff ^ "_A")["_A"].to_dense(inds["lix"], inds["uix"])

            # Take TNs off GPU
            if use_gpu:
                self._gpu_manager.from_gpu([LHS_eff["_A"], RHS_eff["_B"]])

            return LHS_eff_dense, RHS_eff_dense, True, use_gpu

        else:
            # Check if system will fit on GPU
            use_gpu = (
                self._gpu_manager.check_available(
                    0, [n, n], [RHS_eff.dtype, self._x.dtype], LHS_eff["_A"]
                )
                if self._gpu_manager
                else False
            )

            if use_gpu:
                self._gpu_manager.to_gpu((LHS_eff["_A"], RHS_eff["_B"]))

            LHS_eff_op = TNLinearOperator(
                LHS_eff["_A"],
                ldims=inds["ldims"],
                rdims=inds["rdims"],
                left_inds=inds["lix"],
                right_inds=inds["uix"],
            )

            # Compute RHS vector
            RHS_eff_dense = do(
                "reshape", (RHS_eff ^ "_B")["_B"].to_dense(inds["lix"]), (-1, 1)
            )

            # Remove RHS from GPU
            if use_gpu:
                self._gpu_manager.from_gpu([RHS_eff["_B"]])

            return LHS_eff_op, RHS_eff_dense, False, use_gpu

    def _right_canonize_x(self):
        for i in reversed(range(1, self._order)):
            # Get core and right matricize
            xeff, shape, lix, uix = self._right_matricize(i, self._x[i], self._x)
            xeff = do("transpose", xeff)

            # Run QR
            # xeff, rv = do("linalg.qr", xeff)
            xeff, rv = qr(xeff, mode="economic")
            xeff = do("transpose", xeff)

            # Place back into core
            shape[0] = xeff.shape[0]
            xeff = self._right_tensorize(i, xeff, self._x, shape, lix, uix)
            self._x[i].modify(data=xeff.data)
            self._xH[i].modify(data=xeff.conj().data)

            # Compute left tensor orthogonalization
            xeff, shape, lix, uix = self._left_matricize(i - 1, self._x[i - 1], self._x)

            xeff = xeff @ do("transpose", rv)

            # Place back into left core
            shape[-1] = xeff.shape[-1]
            xeff = self._left_tensorize(i - 1, xeff, self._x, shape, lix, uix)
            self._x[i - 1].modify(data=xeff.data)
            self._xH[i - 1].modify(data=xeff.conj().data)

    def _right_canonize_z(self):
        """"""
        for i in reversed(range(1, self._order)):
            self._right_canonize_z_site(i, do("reshape", self._x[i].data, (-1, 1)))

    def _right_canonize_z_site(self, i, xeff):
        use_gpu = False

        if self._sweep_num > 0:
            # Move environments and get TNs
            self._zLHS_ME.move_to(i)
            self._zRHS_ME.move_to(i)

            # Get index pointers for residual projections
            inds = {
                "ldims": list(self._zH[i].shape),
                "lix": list(self._zH[i].inds),
                "rdims": list(self._x[i].shape),
                "uix": list(self._x[i].inds),
            }

            if i < self._order - 1:
                inds["ldims"][-1] = self._zH[i + 1].shape[0]

            # Form local operators for residual system
            Aeff, Beff, dense, use_gpu = self._form_local_ops(
                self._zLHS_ME(), self._zRHS_ME(), inds
            )

            # Ensure xeff is on the correct device
            if use_gpu:
                if not isinstance(xeff, cu.ndarray):
                    xeff = self._gpu_manager.to_gpu(xeff)

            else:
                if isinstance(xeff, cu.ndarray):
                    xeff = self._gpu_manager.from_gpu(xeff)

            # Compute residual
            zeff = Beff - Aeff @ xeff

            # Reshape zeff for SVD
            zeff = Tensor(do("reshape", zeff, inds["ldims"]), inds=inds["lix"])
            zeff, shape, lix, uix = self._right_matricize(i, zeff, self._zH)

            # Run SVD if we're truncating
            _, _, zeff = self._svd(zeff, use_gpu)
            # _, _, zeff = do("linalg.svd", zeff)
            zeff = do("transpose", zeff)

            # Truncate
            r = min(self._kickrank, zeff.shape[-1])

            zeff = do("conjugate", zeff[:, :r])

            # Enrich residual
            if i < self._order and self._kickrank2 > 0:
                zeff = do(
                    "concatenate",
                    [
                        zeff,
                        do(
                            "random.normal",
                            size=(zeff.shape[0], self._kickrank2),
                            like=zeff,
                        ),
                    ],
                    axis=-1,
                    like=zeff,
                )

            # Clean GPU from extra operators
            if use_gpu:
                self._gpu_manager.from_gpu([Beff])

                if not dense:
                    self._gpu_manager.from_gpu([self._zLHS_ME()["_A"]])

        else:
            zeff, shape, lix, uix = self._right_matricize(i, self._zH[i], self._zH)
            zeff = do("transpose", zeff)

        # zeff, _ = do("linalg.qr", zeff)
        zeff, _ = qr(zeff, mode="economic")
        zeff = do("transpose", zeff)

        shape[0] = zeff.shape[0]
        zeff = self._right_tensorize(i, zeff, self._zH, shape, lix, uix)

        # Clean GPU from extra operators
        if use_gpu:
            xeff, zeff = self._gpu_manager.from_gpu([xeff, zeff])

        self._zH[i].modify(data=zeff.data)

    def _left_canonize_z_site(self, i, xeff):
        use_gpu = False

        if self._sweep_num > 0:
            # Move environments and get TNs
            self._zLHS_ME.move_to(i)
            self._zRHS_ME.move_to(i)

            # Get index pointers for residual projections
            inds = {
                "ldims": list(self._zH[i].shape),
                "lix": list(self._zH[i].inds),
                "rdims": list(self._x[i].shape),
                "uix": list(self._x[i].inds),
            }

            # Correct for new rank
            if i > 0:
                inds["ldims"][0] = self._zLHS_ME()["_LEFT"].shape[0]
            if i < self._order - 1:
                inds["ldims"][-1] = self._zLHS_ME()["_RIGHT"].shape[0]

            # Form local operators for residual system
            Aeff, Beff, dense, use_gpu = self._form_local_ops(
                self._zLHS_ME(), self._zRHS_ME(), inds
            )

            # Ensure xeff is on the correct device
            if use_gpu:
                if not isinstance(xeff, cu.ndarray):
                    xeff = self._gpu_manager.to_gpu(xeff)

            else:
                if isinstance(xeff, cu.ndarray):
                    xeff = self._gpu_manager.from_gpu(xeff)

            # Compute residual
            zeff = Beff - Aeff @ xeff

            # Reshape zeff for SVD
            zeff = Tensor(do("reshape", zeff, inds["ldims"]), inds=inds["lix"])
            zeff, shape, lix, uix = self._left_matricize(i, zeff, self._zH)

            # Run SVD if we're truncating
            zeff, _, _ = self._svd(zeff, use_gpu)
            # zeff, _, _ = do("linalg.svd", zeff)

            # Truncate
            r = min(self._kickrank, zeff.shape[-1])
            zeff = zeff[:, : min(self._kickrank, zeff.shape[-1])]

            # Enrich residual
            if i < self._order - 1 and self._kickrank2 > 0:
                zeff = do(
                    "concatenate",
                    [
                        zeff,
                        do(
                            "random.normal",
                            size=(zeff.shape[0], self._kickrank2),
                            like=zeff,
                        ),
                    ],
                    like=zeff,
                )

            # Clean GPU from extra operators
            if use_gpu:
                self._gpu_manager.from_gpu([Beff])

                if not dense:
                    self._gpu_manager.from_gpu([self._zLHS_ME()["_A"]])

        else:
            zeff, shape, lix, uix = self._left_matricize(i, self._zH[i], self._zH)

        # Run QR orthogonalization
        # zeff, _ = do("linalg.qr", zeff)
        zeff, _ = qr(zeff, mode="economic")

        if i < self._order - 1:
            shape[-1] = r

        # Tensorize
        zeff = self._left_tensorize(i, zeff[:, :r], self._zH, shape, lix, uix)

        # Clean GPU from extra operators
        if use_gpu:
            xeff, zeff = self._gpu_manager.from_gpu([xeff, zeff])

        self._zH[i].modify(data=zeff.data)

    # =============================================================
    # Matricization and tensorization methods

    def _left_matricize(self, i, x, tn):
        """
        Left matricization to shape (r_left * i, r_right) where i is the physical index
        and r are the ranks.

        Also known as left unfolding.
        """
        if i < tn.L - 1:
            lix = []
            shape = []

            # Add left rank if applicable
            if i > 0:
                lix += [tn.bond(i - 1, i)]
                shape += [x.ind_size(lix[-1])]

            # Add physical index
            lix += [tn[i].inds[-2]]
            shape += [x.ind_size(lix[-1])]

            # Right rank
            uix = [tn.bond(i, i + 1)]
            shape += [x.ind_size(uix[-1])]

            # Matricize to shape (r_left * i, r_right)
            x = x.to_dense(lix, uix)
            return x, shape, lix, uix

        else:
            # Matricize to shape (r_left * i, 1)
            lix = x.inds
            shape = [x.ind_size(l) for l in lix]
            x = do("reshape", x.to_dense(lix), (-1, 1))
            return x, shape, lix, ()

    def _right_matricize(self, i, x, tn):
        """
        Right matricization to shape (r_left, i * r_right) where i is the physical index
        and r are the ranks.

        Also known as right unfolding.
        """
        if i > 0:
            # Left rank
            lix = [tn.bond(i - 1, i)]
            shape = [x.ind_size(lix[-1])]

            # Physical index
            uix = [tn[i].inds[1]]
            shape += [x.ind_size(uix[-1])]

            # Right rank
            if i < tn.L - 1:
                uix += [tn.bond(i, i + 1)]
                shape += [tn[i + 1].ind_size(uix[-1])]

            # Matricize to shape (r_left, i * r_right)
            x = x.to_dense(lix, uix)
            return x, shape, lix, uix

        else:
            # Matricize to shape (1, i * r_right)
            uix = x.inds
            shape = [x.ind_size(r) for r in uix]
            x = do("reshape", x.to_dense(uix), (1, -1))
            return x, shape, (), uix

    def _left_tensorize(self, i, x, tn, shape, lix, uix):
        """Tensorization or folding of left matricized core."""
        x = (
            Tensor(do("reshape", x, shape), inds=(*lix, *uix))
            if i < tn.L - 1
            else Tensor(do("reshape", do("reshape", x, (-1)), shape), inds=lix)
        )
        x.transpose_(*tn[i].inds)
        return x

    def _right_tensorize(self, i, x, tn, shape, lix, uix):
        """Tensorization or folding of right matricized core."""
        x = (
            Tensor(do("reshape", x, shape), inds=(*lix, *uix))
            if i > 0
            else Tensor(do("reshape", do("reshape", x, (-1)), shape), inds=uix)
        )
        x.transpose_(*tn[i].inds)
        return x

    # =============================================================
    # Static methods

    @staticmethod
    def _set_seq(seq):
        seq = (seq,) if isinstance(seq, Union[int, float]) else tuple(seq)
        return seq[0], itertools.chain(seq, itertools.repeat(seq[-1]))

    @staticmethod
    def _svd(A, use_gpu):
        if use_gpu:
            return cu_svd(A, full_matrices=False)
        else:
            return sc_svd(A, full_matrices=False, lapack_driver="gesvd")

    @staticmethod
    def _gmres(A, b, rtol, restart, maxiter, use_gpu):
        """Solve linear system using Scipy or Cupy GMRES depending on GPU
        availability."""
        if use_gpu:
            return cu_gmres(
                A=A,
                b=b,
                tol=rtol,
                restart=restart,
                maxiter=maxiter,
            )

        else:
            return sc_gmres(
                A=A,
                b=b,
                rtol=rtol,
                restart=restart,
                maxiter=maxiter,
            )

    # =============================================================
    # Print methods

    def _print_pre_sweep(self, i):
        if self._verbose > 0:
            print(f"Sweep {i + 1},", flush=True, end=" ")

    def _print_post_sweep(self):
        if self._verbose:
            print(
                "max_dx: {}, max_res: {}, max_rank: {}/{}".format(
                    self._max_dx,
                    self._max_res,
                    np.max(self._x.bond_sizes()),
                    self._max_rank,
                )
            )

    # =============================================================
    # Getters

    @property
    def x(self):
        # copy = self._x.copy()
        copy = MatrixProductState([core.data for core in self._x], shape="lpr")
        # for i in range(self._order):
        #     if i == 0:
        #         copy[i].modify(inds=(self._inds[i], copy[i].inds[-1]))
        #     elif i == self._order - 1:
        #         copy[i].modify(inds=(copy[i].inds[0], self._inds[i]))
        #     else:
        #         copy[i].modify(
        #             inds=(copy[i].inds[0], self._inds[i], copy[i].inds[-1]),
        #         )
        #
        # copy.drop_tags("_x")
        return copy

    @property
    def tol(self):
        return self._tol

    @property
    def max_res(self):
        return self._max_res

    @property
    def op_type(self):
        return MatrixProductOperator
