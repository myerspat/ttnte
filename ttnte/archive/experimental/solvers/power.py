import warnings
import random

from quimb.tensor import tensor_network_apply_op_vec, MPS_rand_state


class Power(object):
    def __init__(
        self,
        lhs,
        F,
        rhs=lambda F, psi, k: tensor_network_apply_op_vec(F, psi, contract=True) / k,
        verbose=False,
        print_every=5,
    ):
        """"""
        self._lhs = lhs
        self._F = F
        self._rhs = rhs

        self._psi = None
        self._k = None
        self._num_iter = None
        self._loss = None
        self._tol = None
        self._verbose = verbose
        self._print_every = print_every

    # =======================================================================
    # Methods

    def solve(
        self,
        linear_solver,
        tol=1e-5,
        max_iter=100,
        psi0=None,
        k0=None,
        rank=None,
        callback=None,
        **linear_solver_opts,
    ):
        """"""
        # Get initial guess
        if psi0 is None:
            assert rank is not None
            psi0 = MPS_rand_state(
                L=self._F.L,
                bond_dim=rank,
                phys_dim=[self._F.phys_dim(site=site) for site in self._F.sites],
                dist="uniform",
                dtype=self._lhs.dtype,
            )
            psi0 /= psi0.norm()
        if k0 is None:
            k0 = random.random()

        self._psi0 = psi0.copy()
        self._k0 = k0

        # Copy for initial
        self._psi = psi0.copy()
        self._k = k0
        self._tol = tol
        loss0 = 1

        norm0 = tensor_network_apply_op_vec(self._F, psi0).norm()

        for self._num_iter in range(max_iter):
            # Compute x in Ax=b
            self._psi = linear_solver.solve(
                A=self._lhs,
                B=self._rhs(self._F, psi0, k0),
                x0=psi0,
                **linear_solver_opts,
            )

            norm = tensor_network_apply_op_vec(self._F, self._psi).norm()
            self._k = k0 * norm / norm0
            self._loss = abs((self._psi - psi0).norm() / psi0.norm())

            if self._verbose and self._num_iter % self._print_every == 0:
                print(
                    f"--   Iteration = {self._num_iter},"
                    + f" k = {self._k},"
                    + f" psi error = {self._loss}"
                )

            # Run callback if given
            if callback:
                callback(self)

            # Return if tolerance is met
            if self._loss < self._tol:
                if self._verbose:
                    print(f"-- Converged: k = {round(self._k, 8)}")

                # Normalize eigenvector
                self._psi /= self._psi.norm()

                return self._k, self._psi

            elif self._loss > loss0 and self._num_iter > 2:
                if self._verbose:
                    warnings.warn(
                        f"Iteration {self._num_iter} error is greater "
                        + f"than iteration {self._num_iter - 1}",
                        RuntimeWarning,
                    )

            psi0 = self._psi.copy()
            k0 = self._k
            loss0 = self._loss
            norm0 = norm

    # =======================================================================
    # Getters

    @property
    def k(self):
        return self._k

    @property
    def psi(self):
        return self._psi

    @property
    def loss(self):
        return self._loss

    @property
    def converged(self):
        assert self._loss is not None and self._tol is not None
        return self._loss < self._tol

    @property
    def num_iter(self):
        return self._num_iter
