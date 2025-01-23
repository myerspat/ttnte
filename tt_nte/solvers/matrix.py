import numpy as np
from scipy.sparse.linalg import eigs, inv

from tt_nte.solvers._base import Solver


class Matrix(Solver):
    def __init__(self, method, verbose=False, tt_driver="scikit_tt"):
        """Matrix solver for given method."""
        # Initialize base class
        super().__init__(
            method.H.matricize(tt_driver),
            method.F.matricize(tt_driver),
            method.S.matricize(tt_driver),
            verbose=verbose,
        )

    # =======================================================================
    # Methods

    def ges(self):
        """Generalized eigenvalue solver using scipy.sparse.linalg.eigs()."""
        self._setup()

        def solver(A, B):
            l, v = eigs(A, 1, B)
            return l.real[0], v.real.reshape((-1, 1))

        def norm(x, p):
            if p == 1:
                return np.sum(x)
            else:
                return np.linalg.norm(x)

        super()._ges(solver=solver, norm=norm)

    def power(self, tol=1e-6, max_iter=100, k0=None, psi0=None):
        """Power iteration with H inversion using scipy.sparse.linalg.inv()."""
        if k0 is None and psi0 is None:
            k0, psi0 = self._setup()

        A_inv = inv(self._M)

        def norm(x, p):
            if p == 1:
                return np.sum(x)
            else:
                return np.linalg.norm(x)

        super()._power(
            psi0=psi0,
            k0=k0,
            solver=lambda A, B, x0: A_inv.dot(B.dot(x0)),
            norm=norm,
            matvec=lambda A, x: A @ x,
            tol=tol,
            max_iter=max_iter,
        )

    def _setup(self):
        """Create initial guess for psi and k."""
        # Initial guess for psi and k
        psi0 = np.random.rand(self._M.shape[1]).reshape((-1, 1))
        psi0 *= 1 / np.linalg.norm(psi0, 2)
        k0 = np.random.rand(1)[0]

        return k0, psi0

    # =======================================================================
    # Getters

    @property
    def psi(self):
        return super().psi.flatten()
