import numpy as np
import tt
from tt.amen import amen_solve

from tt_nte.solvers._base import Solver
from tt_nte.tensor_train import TensorTrain


class AMEn(Solver):
    def __init__(self, method, verbose=False):
        """Eigenvalue solver using the Alternating Minimal Energy Method to solve
        Ax=b."""
        # Initialize base class
        super().__init__(
            method.H.train("ttpy"),
            method.F.train("ttpy"),
            method.S.train("ttpy"),
            verbose,
        )

    # =======================================================================
    # Methods

    def power(
        self,
        ranks=None,
        tol=1e-6,
        max_iter=100,
        amen_tol=1e-6,
        amen_nswp=20,
        amen_kickrank=4,
        k0=None,
        psi0=None,
    ):
        """
        Power iteration using tt.amen.amen_solve().

        ``amen_tol`` controls the
        tolerance of the solution produced by tt.amen.amen_solve(). ``ranks``
        controls the ranks of the cores in the solution.
        """
        # Setup power iteration
        if k0 is None and psi0 is None:
            psi0, k0 = self._setup(ranks)
        elif isinstance(psi0, TensorTrain):
            psi0 = psi0.train("ttpy")

        assert psi0 is not None and k0 is not None

        def solver(A, B, x0):
            return amen_solve(
                A,
                tt.matvec(B, x0),
                x0,
                amen_tol,
                kickrank=amen_kickrank,
                verb=self._verbose,
                nswp=amen_nswp,
            )

        super()._power(
            psi0=psi0,
            k0=k0,
            solver=solver,
            norm=lambda x, p: x.norm(),
            matvec=tt.matvec,
            tol=tol,
            max_iter=max_iter,
        )

    def _setup(self, ranks):
        # Get maximum ranks for each core
        ranks = (
            ((np.array([self._M.r, self._F.r])).max(axis=0).tolist())
            if ranks is None
            else ranks
        )

        # Initial guess for psi and k
        psi0 = tt.vector.from_list(
            TensorTrain.rand(self._F.n, [1] * self._F.d, ranks).cores
        )
        k0 = np.random.rand(1)[0]

        return psi0, k0

    # =======================================================================
    # Getters

    @property
    def psi(self):
        return TensorTrain(tt.vector.to_list(super().psi))
