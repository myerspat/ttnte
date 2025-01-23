import numpy as np
import scikit_tt.solvers.evp as evp
import scikit_tt.solvers.sle as sle

from tt_nte.solvers._base import Solver
from tt_nte.tensor_train import TensorTrain


class ALS(Solver):
    def __init__(self, method, verbose=False):
        """Eignenvalue solver using Alternating Linear Scheme (ALS) to solve Ax=b."""
        # Initialize base class
        super().__init__(
            method.H.train("scikit_tt"),
            method.F.train("scikit_tt"),
            method.S.train("scikit_tt"),
            verbose,
        )

    # =======================================================================
    # Methods

    def ges(self, ranks=None, repeats=10):
        """Generalized eigenvalue solver using scikit_tt.solvers.evp.als()."""
        # Setup power iteration
        psi0, _ = self._setup(ranks)

        def solver(A, B):
            l, v, _ = evp.als(A, psi0, operator_gevp=B, repeats=repeats)
            return l, v

        super()._ges(
            solver=solver,
            norm=lambda x, p: x.norm(p),
        )

    def power(self, ranks=None, tol=1e-6, max_iter=100, k0=None, psi0=None):
        """
        Power iteration using scikit_tt.solvers.sle.als().

        ``ranks`` is the
        ranks of the cores in the solution.
        """
        # Setup power iteration
        if k0 is None and psi0 is None:
            psi0, k0 = self._setup(ranks)
        elif isinstance(psi0, TensorTrain):
            psi0 = psi0.train("scikit_tt")

        super()._power(
            psi0=psi0,
            k0=k0,
            solver=lambda A, B, x0: sle.als(A, x0, B.dot(x0)),
            norm=lambda x, p: x.norm(p),
            matvec=lambda A, x: A @ x,
            tol=tol,
            max_iter=max_iter,
        )

    def _setup(self, ranks):
        # Get maximum ranks for each core
        ranks = (
            ((np.array([self._M.ranks, self._F.ranks])).max(axis=0).tolist())
            if ranks is None
            else ranks
        )

        # Initial guess for psi and k
        psi0 = TensorTrain.rand(self._F.row_dims, [1] * self._F.order, ranks).train(
            "scikit_tt"
        )
        k0 = np.random.rand(1)[0]

        return psi0, k0

    # =======================================================================
    # Getters

    @property
    def psi(self):
        return TensorTrain(super().psi.cores)
