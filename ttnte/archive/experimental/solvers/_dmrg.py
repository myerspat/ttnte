import itertools
from typing import Union, Optional

import numpy as np
import cupy as cu
from autoray import do
from quimb.tensor import (
    MatrixProductState,
    MatrixProductOperator,
    MPS_rand_state,
    tensor_network_apply_op_vec,
    TensorNetwork,
)
from quimb.tensor.tensor_1d import TensorNetwork1DFlat
from quimb.tensor.tensor_core import TNLinearOperator, Tensor
from quimb.core import prod
from quimb.linalg.base_linalg import eigh


class DMRG(object):
    def __init__(
        self,
        num_sites: int,
        verbose: bool = False,
        **opts,
    ):
        """
        Implementation of alternating linear system (ALS) and modified-ALS (MALS), also
        known as 1-site and 2-site DMRG.

        This implementation mirrors that of quimb for linear systems of equations.
        """
        self._num_sites = num_sites
        self._verbose = verbose

        self._tol = None
        self._loss = None
        self._l = None
        self._l0 = None

        # Base solver options
        self._use_gpu = opts.get("use_gpu", False)
        self._min_system_size = opts.get("min_system_size", 1000)
        self._rand_strength = opts.get("rand_strength", 1e-6)
        self._which = opts.get("which", "SA")
        self._local_eig_dense = opts.get("local_eig_dense", None)
        self._local_eig_backend = opts.get("local_eig_backend", None)
        self._local_eig_EPSType = opts.get("local_eig_EPSType", None)
        self._local_eig_ncv = opts.get("local_eig_ncv", 4)
        self._local_eig_tol = opts.get("local_eig_tol", 1e-3)
        self._local_eig_maxiter = opts.get("local_eig_maxiter", None)

    # =============================================================
    # Methods

    def solve(
        self,
        A: MatrixProductOperator,
        B: Optional[Union[MatrixProductOperator, MatrixProductState]],
        x0: Optional[MatrixProductState],
        tols=1e-5,
        max_ranks=None,
        cutoffs=1e-9,
        sweep_sequence: str = "RL",
        max_sweeps: int = 1,
        method: str = "svd",
        cutoff_mode: str = "rel",
        virtual: bool = True,
        verbose: Optional[bool] = None,
    ):
        """"""
        self._verbose = verbose if verbose else self._verbose

        # Determine initial rank, tol and sequences
        max_rank, max_ranks = self._set_seq(
            (max_ranks if max_ranks is not None else self._max_rank(A))
        )
        _, tols = self._set_seq(tols)
        _, cutoffs = self._set_seq(cutoffs)

        # Setup DMRG solver
        self._setup(A=A, B=B, x0=x0, rank=max_rank, virtual=virtual)

        # Begin sweep iterations
        previous_direction = "0"
        for i in range(max_sweeps):
            # Get next ranks and tolerances
            max_rank, self._tol, cutoff = (next(max_ranks), next(tols), next(cutoffs))

            # Update rankd of 1-site DMRG manually
            if self._num_sites == 1:
                self._x.expand_bond_dimension(
                    max_rank, bra=self._xH, rand_strength=self._rand_strength
                )
                self._copy_x()

            for direction in sweep_sequence:
                self._print_pre_sweep(i, direction, max_rank, self._verbose)

                # Run left or right sweeps
                self._sweep(
                    direction=direction,
                    canonize=(
                        False
                        if direction + previous_direction in ("RL", "LR")
                        else True
                    ),
                    max_bond=max_rank,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    method=method,
                    verbose=self._verbose,
                )

                # Evaluate convergance
                self._loss = self._eval_loss()
                self._print_loss(i, direction, self._verbose)
                if self._loss < self._tol:
                    return self.x

                previous_direction = direction

        return self.x

    def _setup(
        self,
        A: MatrixProductOperator,
        B: Optional[Union[MatrixProductOperator, MatrixProductState]],
        x0: Optional[MatrixProductState],
        rank: Optional[int],
        virtual: bool = True,
    ):
        """"""
        # Get LHS operator and its properties
        self._A = A.copy(virtual=virtual)
        self._A.add_tag("_A")
        self._order = self._A.L
        self._phys_dims = [self._A.phys_dim(site=site) for site in self._A.sites]

        # Get RHS operator/vector/None
        self._eig_mode = not isinstance(B, MatrixProductState)
        self._use_gpu = False if self._eig_mode else self._use_gpu

        # Initial guess
        self._x = (
            x0.copy()
            if x0
            else MPS_rand_state(
                L=self._order,
                bond_dim=rank,
                phys_dim=self._phys_dims,
                dist="uniform",
            )
        )

        # Conjugate transpose
        self._xH = self._x.H

        # Add tags
        self._x.add_tag("_x")
        self._xH.add_tag("_xH")

        # Align system and combine to LHS
        self._x.align_(self._A, self._xH)
        self._LHS = self._xH | self._A | self._x

        # Copies
        self._B = None
        self._x_cpy = None
        self._xH_cpy = None
        self._RHS = None

        # Determine linear or eigenvalue problem
        if B:
            self._B = B.copy(virtual=virtual)
            self._B.add_tag("_B")

            # Make virtual copy of conjugate transpose
            self._xH_cpy = self._xH.copy(virtual=True)

            if self._eig_mode:
                # Make virtual copy of x, align, and initialize RHS
                self._x_cpy = self._x.copy(virtual=True)
                self._x_cpy.align_(self._B, self._xH_cpy)
                self._RHS = self._xH_cpy | self._B | self._x_cpy

            else:
                # Align linear system and initialize right hand side
                self._B.align_(self._xH_cpy)
                self._RHS = self._xH_cpy | self._B

    def _copy_x(self):
        if self._x_cpy:
            for i in range(self._order):
                self._x_cpy[i].modify(data=self._x[i].data)
                assert self._x_cpy[i].shape == self._x[i].shape

        if self._xH_cpy:
            for i in range(self._order):
                self._xH_cpy[i].modify(data=self._xH[i].data)
                assert self._xH_cpy[i].shape == self._xH[i].shape

    def _eval_loss(self):
        """"""
        if self._eig_mode:
            self._l = self._LHS ^ all
            self._l /= self._RHS ^ all if self._RHS else 1

            loss = abs(self._l - self._l0) if self._l0 else self._tol + 1

            self._l0 = self._l

            return loss

        elif self._use_gpu:
            self._A = self._numpy2cupy(self._A)
            self._x = self._numpy2cupy(self._x)
            self._B = self._numpy2cupy(self._B)

            loss = abs(
                1 / 2 * tensor_network_apply_op_vec(self._A, self._x).H @ self._x
                - self._B.H @ self._x
            )

            # loss = abs(
            #     tensor_network_apply_op_vec(self._A.H, self._x.H, which_A="upper")
            #     @ tensor_network_apply_op_vec(self._A, self._x)
            #     - 2 * self._B.H @ tensor_network_apply_op_vec(self._A, self._x)
            #     + self._B.H @ self._B
            # )

            self._A = self._cupy2numpy(self._A)
            self._x = self._cupy2numpy(self._x)
            self._B = self._cupy2numpy(self._B)

            cu.get_default_memory_pool().free_all_blocks()

            return loss

        else:
            return abs(
                1 / 2 * tensor_network_apply_op_vec(self._A, self._x).H @ self._x
                - self._B.H @ self._x
            )

    # =============================================================
    # Sweep methods

    def _sweep(self, direction, canonize, verbose, **compress_opts):
        """"""
        # Orthogonalize
        if canonize:
            {"R": self._x.right_canonize, "L": self._x.left_canonize}[direction](
                bra=self._xH, inplace=True
            )
            self._copy_x()

        # Define sweeping variables
        direction, begin, sweep = {
            "R": ("right", "left", range(0, self._order - self._num_sites + 1)),
            "L": ("left", "right", range(self._order - self._num_sites, -1, -1)),
        }[direction]

        # Setup local problems
        self._LHS_ME = MovingEnvironment(
            tn=self._LHS, begin=begin, num_sites=self._num_sites
        )
        if self._RHS:
            self._RHS_ME = MovingEnvironment(
                tn=self._RHS, begin=begin, num_sites=self._num_sites
            )

        # Perform sweep
        for i in sweep:
            self._update_local_x(i=i, direction=direction, **compress_opts)

    def _update_local_x(self, i, direction, **compress_opts):
        """"""
        # Moving environment
        self._LHS_ME.move_to(i)
        if self._RHS:
            self._RHS_ME.move_to(i)

        # Evaluate local problem
        {
            1: self._update_local_x_1site,
            2: self._update_local_x_2site,
        }[
            self._num_sites
        ](i, direction, **compress_opts)

    def _update_local_x_1site(self, i, direction, **compress_opts):
        """"""
        # Get index pointers
        inds = {
            "dims": self._x[i].shape,
            "uix": self._x[i].inds,
            "lix": self._xH[i].inds,
        }
        if self._x_cpy:
            inds["lix_cpy"] = self._x_cpy[i].inds
        if self._xH_cpy:
            inds["lix_cpy"] = self._xH_cpy[i].inds

        # Form local operators
        Aeff, Beff, gpu = self._form_local_ops(i, **inds)

        # Solve system
        xeff = None
        if self._eig_mode:
            raise NotImplementedError()
        else:
            xeff = do("linalg.solve", Aeff, Beff)

        # Insert updated xeff back into networks
        xeff = xeff.get() if gpu else xeff
        xeff = xeff.reshape(self._x[i].shape)
        self._x[i].modify(data=xeff)
        self._xH[i].modify(data=xeff.conj())

        # Orthogonalize
        self._canonize_after_1site_update(direction, i)

        if self._x_cpy:
            self._x_cpy[i].modify(data=self._x[i].data)
        if self._xH_cpy:
            self._xH_cpy[i].modify(data=self._xH[i].data)

    def _update_local_x_2site(self, i, direction, **compress_opts):
        """"""
        # Get index pointers
        u_ind_names = ["dims", "uix_L", "uix_R", "uix", "u_bond_ind"]
        l_ind_names = ["lix_L", "lix_R", "lix", "l_bond_ind"]
        ind_names = u_ind_names + l_ind_names
        ind_names += [name + "_cpy" for name in u_ind_names] if self._x_cpy else []
        ind_names += [name + "_cpy" for name in l_ind_names] if self._xH_cpy else []
        inds = {
            name: ind
            for name, ind in zip(
                ind_names,
                self._parse_2site_inds_dims(i, self._x, self._xH)
                + self._parse_2site_inds_dims(i, self._x_cpy, self._xH_cpy),
            )
        }

        # Form local operators
        Aeff, Beff, gpu = self._form_local_ops(i, **inds)

        # Solve system
        xeff = None
        if self._eig_mode:
            _, xeff = self._eigs(Aeff, B=Beff, v0=self._x[i].data.ravel())
        else:
            xeff = do("linalg.solve", Aeff, Beff)

        # Split the two site local solution
        xeff = xeff.get() if gpu else xeff
        xeff = Tensor(xeff.reshape(inds["dims"]), inds["uix"])
        L, R = xeff.split(
            left_inds=inds["uix_L"],
            get="arrays",
            absorb=direction,
            right_inds=inds["uix_R"],
            **compress_opts,
        )

        # Insert back into networks
        self._x[i].modify(data=L, inds=(*inds["uix_L"], inds["u_bond_ind"]))
        self._x[i + 1].modify(data=R, inds=(inds["u_bond_ind"], *inds["uix_R"]))
        self._xH[i].modify(data=L.conj(), inds=(*inds["lix_L"], inds["l_bond_ind"]))
        self._xH[i + 1].modify(data=R.conj(), inds=(inds["l_bond_ind"], *inds["lix_R"]))

        if self._x_cpy:
            self._x_cpy[i].modify(
                data=L, inds=(*inds["uix_L_cpy"], inds["u_bond_ind_cpy"])
            )
            self._x_cpy[i + 1].modify(
                data=R, inds=(inds["u_bond_ind_cpy"], *inds["uix_R_cpy"])
            )

        if self._xH_cpy:
            self._xH_cpy[i].modify(
                data=L.conj(), inds=(*inds["lix_L_cpy"], inds["l_bond_ind_cpy"])
            )
            self._xH_cpy[i + 1].modify(
                data=R.conj(), inds=(inds["l_bond_ind_cpy"], *inds["lix_R_cpy"])
            )

    def _form_local_ops(self, i, **inds):
        """"""
        # Get local operators
        LHS_eff = self._LHS_ME()
        RHS_eff = self._RHS_ME() if self._RHS else None

        # GPU flag
        use_gpu = (
            prod(inds["dims"]) > self._min_system_size
            if self._use_gpu is True
            else False
        )

        # Dense flag
        dense = (
            True
            if (
                prod(inds["dims"]) < self._min_system_size
                or use_gpu is True
                or not self._eig_mode
            )
            and self._local_eig_dense is not False
            else False
        )

        if dense:
            LHS_eff_dense = None
            RHS_eff_dense = None

            if use_gpu:
                # Push TN to GPU
                LHS_eff = self._numpy2cupy(LHS_eff)

                # Contract to form matrix operator
                LHS_eff_dense = (LHS_eff ^ "_A")["_A"].to_dense(
                    inds["uix"], inds["lix"]
                )

                # Remove TN from GPU
                LHS_eff = self._cupy2numpy(LHS_eff)

                if RHS_eff is not None:
                    # Push to GPU
                    RHS_eff = self._numpy2cupy(RHS_eff)

                    RHS_eff_dense = (
                        (RHS_eff ^ "_B")["_B"].to_dense(inds["lix_cpy"])
                        if isinstance(self._B, MatrixProductState)
                        else (RHS_eff ^ "_B")["_B"].to_dense(
                            inds["uix_cpy"], inds["lix_cpy"]
                        )
                    )

                    # Remove TN from GPU
                    RHS_eff = self._cupy2numpy(RHS_eff)

            else:
                # Contract to form matrix operator
                LHS_eff_dense = (LHS_eff ^ "_A")["_A"].to_dense(
                    inds["uix"], inds["lix"]
                )
                if RHS_eff is not None:
                    RHS_eff_dense = (
                        (RHS_eff ^ "_B")["_B"].to_dense(inds["lix_cpy"])
                        if isinstance(self._B, MatrixProductState)
                        else (RHS_eff ^ "_B")["_B"].to_dense(
                            inds["uix_cpy"], inds["lix_cpy"]
                        )
                    )

            return LHS_eff_dense, RHS_eff_dense, use_gpu

        else:
            LHS_eff = TNLinearOperator(
                LHS_eff["_A"],
                **{
                    "ldims": inds["dims"],
                    "rdims": inds["dims"],
                    "left_inds": inds["lix"],
                    "right_inds": inds["uix"],
                },
            )
            RHS_eff = (
                TNLinearOperator(
                    RHS_eff["_B"],
                    **{
                        "ldims": inds["dims_cpy"],
                        "rdims": inds["dims_cpy"],
                        "left_inds": inds["lix_cpy"],
                        "right_inds": inds["uix_cpy"],
                    },
                )
                if RHS_eff is not None
                else None
            )
            return LHS_eff, RHS_eff, False

    def _canonize_after_1site_update(self, direction, i):
        """"""
        if (direction == "right") and (i < self._order - 1):
            self._x.left_canonize_site(i, bra=self._xH)
        elif (direction == "left") and (i > 0):
            self._x.right_canonize_site(i, bra=self._xH)

    def _eigs(self, A, B=None, v0=None):
        """Find single eigenpair, using all the internal settings."""
        # intercept generalized eigen
        backend = self._local_eig_backend
        if (backend is None) and (B is not None):
            backend = "LOBPCG"

        return eigh(
            A,
            k=1,
            B=B,
            which=self._which,
            v0=v0,
            backend=backend,
            EPSType=self._local_eig_EPSType,
            ncv=self._local_eig_ncv,
            tol=self._local_eig_tol,
            maxiter=self._local_eig_maxiter,
            fallback_to_scipy=True,
        )

    # =============================================================
    # Static methods

    @staticmethod
    def _max_rank(tn: Union[MatrixProductState, MatrixProductOperator]):
        return int(max(tn.bond_sizes()))

    @staticmethod
    def _set_seq(seq):
        seq = (seq,) if isinstance(seq, Union[int, float]) else tuple(seq)
        return seq[0], itertools.chain(seq, itertools.repeat(seq[-1]))

    @staticmethod
    def _parse_2site_inds_dims(i, x=None, xH=None):
        """Parse dimensions related to two cores used in a local problem in MALS."""
        returns = []
        if x:
            u_bond_ind = x.bond(i, i + 1)
            dims_L, uix_L = zip(
                *((d, ix) for d, ix in zip(x[i].shape, x[i].inds) if ix != u_bond_ind)
            )
            dims_R, uix_R = zip(
                *(
                    (d, ix)
                    for d, ix in zip(x[i + 1].shape, x[i + 1].inds)
                    if ix != u_bond_ind
                )
            )

            returns += [dims_L + dims_R, uix_L, uix_R, uix_L + uix_R, u_bond_ind]

        if xH:
            l_bond_ind = xH.bond(i, i + 1)
            lix_L = tuple(i for i in xH[i].inds if i != l_bond_ind)
            lix_R = tuple(i for i in xH[i + 1].inds if i != l_bond_ind)

            returns += [lix_L, lix_R, lix_L + lix_R, l_bond_ind]

        return returns

    @staticmethod
    def _numpy2cupy(arrays):
        if isinstance(arrays, np.ndarray):
            return cu.array(arrays)

        elif isinstance(arrays, TensorNetwork):
            for core in arrays:
                core.modify(data=cu.array(core.data))

            return arrays

        else:
            for array in arrays:
                array = cu.array(array)

            return arrays

    @staticmethod
    def _cupy2numpy(arrays):
        if isinstance(arrays, np.ndarray):
            return arrays.get()

        elif isinstance(arrays, TensorNetwork):
            for core in arrays:
                core.modify(data=core.data.get())

            return arrays

        else:
            for array in arrays:
                array = array.data.get()

            return arrays

    # =============================================================
    # Print methods

    def _print_pre_sweep(self, i, direction, max_rank, verbose=0):
        if verbose > 0:
            print(
                f"{i + 1}, {direction}, "
                + f"max_rank=({self._max_rank(self._x)}/{max_rank})",
                flush=True,
            )

    def _print_loss(self, i, direction, verbose=0):
        if verbose > 0:
            print(
                f"{i + 1}, {direction}, loss=({self._loss})",
                flush=True,
            )

    # =============================================================
    # Properties

    @property
    def x(self):
        return self._x

    @property
    def converged(self):
        if self._loss is not None and self._tol is not None:
            return self._loss < self._tol
        else:
            raise RuntimeError("No loss and tolerance are known")


# ----------------------------Moving Environment---------------------------- #


class MovingEnvironment(object):
    def __init__(
        self,
        tn: TensorNetwork1DFlat,
        begin: str,
        num_sites: int,
    ):
        """Environment to define the Tensor objects associated with each local
        problem."""
        # Copy args
        self._tn = tn.copy(virtual=True)
        self._begin = begin
        self._num_sites = num_sites

        # Get order and site tag formatter
        self._order = tn.L
        self._site_tag = lambda i: tn.site_tag_id.format(i % self._order)

        # Initialize local problems
        self._setup(begin, 0, self._order - num_sites + 1)

    # =============================================================
    # Methods

    def _setup(self, begin, start, stop):
        """"""
        self._segment = range(start, stop)

        # Create virtual copy
        tnc = self._tn.copy(virtual=True)

        # Generate dummy left and right envs
        tnc |= Tensor(tags="_LEFT").astype(self._tn.dtype)
        tnc |= Tensor(tags="_RIGHT").astype(self._tn.dtype)

        if begin == "left":
            tags_initial = ["_RIGHT"] + [
                self._site_tag(stop - 1 + b) for b in range(self._num_sites)
            ]
            self._envs = {stop - 1: tnc.select_any(tags_initial)}

            for i in reversed(range(start, stop - 1)):
                # Add a new site to previous environment,
                # and contract one site
                self._envs[i] = self._envs[i + 1].copy(virtual=True)
                self._envs[i] |= tnc.select(i)
                self._envs[i] ^= ("_RIGHT", self._site_tag(i + self._num_sites))

            self._envs[start] |= tnc["_LEFT"]
            self._pos = start

        elif begin == "right":
            tags_initial = ["_LEFT"] + [
                self._site_tag(start + b) for b in range(self._num_sites)
            ]
            self._envs = {start: tnc.select_any(tags_initial)}

            for i in range(start + 1, stop):
                # Add a new site to previous environment,
                # and contract one site
                self._envs[i] = self._envs[i - 1].copy(virtual=True)
                self._envs[i] |= tnc.select(i + self._num_sites - 1)
                self._envs[i] ^= ("_LEFT", self._site_tag(i - 1))

            self._envs[stop - 1] |= tnc["_RIGHT"]
            self._pos = stop - 1

        else:
            raise ValueError("``begin`` must be 'left' or 'right'")

    def move_right(self):
        """"""
        self._pos = (self._pos + 1) % self._order

        if self._pos >= self._segment.start + 1:
            # Insert the updated left environment from previous step
            # Contract left environment with updated site just to left
            new_left = self._envs[self._pos - 1].select(
                ["_LEFT", self._site_tag(self._pos - 1)], which="any"
            )
            self._envs[self._pos] |= new_left ^ all

    def move_left(self):
        """"""
        self._pos = (self._pos - 1) % self._order

        if self._pos <= self._segment.stop - 2:
            # Insert the updated right environment from previous step
            # Contract right environment with updated site just to right
            new_right = self._envs[self._pos + 1].select(
                ["_RIGHT", self._site_tag(self._pos + self._num_sites)], which="any"
            )
            self._envs[self._pos] |= new_right ^ all

    def move_to(self, i):
        """"""
        direction = "left" if i < self._pos else "right"

        while self._pos != i % self._order:
            {"left": self.move_left, "right": self.move_right}[direction]()

    def __call__(self):
        """"""
        return self._envs[self._pos]
