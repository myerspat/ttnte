import time
from typing import Optional, Tuple, Union

import numpy as np
import torch as tn
from torchtt._iterative_solvers import gmres_restart

from .linear_operator import LinearOperator


def fixed_source(
    LHS: LinearOperator,
    RHS: Union[Tuple[LinearOperator], LinearOperator],
    device: Optional[int] = None,
    tols: Optional[Union[Tuple[float], float]] = 1e-8,
    max_iters: Optional[Union[Tuple[int], int]] = 500,
    linear_solver_opts: dict = {
        "max_iterations": 100,
        "threshold": 1e-10,
        "resets": 5,
    },
):
    """
    Compute largest eigenvalue and corresponding eigenvector of :math:`Ax=\\lambda B_1 x
    + B_2x + ...`. This solver uses power iteration with GMRES for the resulting linear
    system.

    Parameters
    ----------
    LHS: ttnte.linalg.LinearOperator
        Left hand side operator.
    RHS: tuple of ttnte.linalg.LinearOperator
        Right hand side operators.
    tols: tuple of float, default=(1e-8,)
        Tolerances for solvers of each ``RHS`` operator.
    max_iters: tuple of int, default=(500,)
        Maximum number of iterations for each ``RHS`` operator.
    device: int, default=None
        Device to compute problem on.
    linear_solver_opts: dict
        GMRES options including ``max_threshold``, ``threshold``, and ``resets``.
    """
    # Convert singles to tuples
    RHS = (RHS,) if isinstance(RHS, LinearOperator) else RHS
    tols = (tols,) if isinstance(tols, float) else tols
    max_iters = (max_iters,) if isinstance(max_iters, int) else max_iters

    # Check shapes of operators
    assert (
        isinstance(RHS, tuple)
        and np.array([LHS.shape == RHS[i].shape for i in range(len(RHS))]).all()
    )

    # Send operators to GPU
    if device is not None:
        LHS = LHS.cuda(device)
        RHS = tuple([RHS[i].cuda(device) for i in range(len(RHS))])

    # Create initial guess
    psi0 = tn.ones(LHS.M, device=device)
    psi = None

    print("Starting power iteration")
    error = np.inf
    start = time.time()
    # Run GMRES
    psi = gmres_restart(
        LinOp=LHS,
        b=(RHS[0] @ psi0).reshape((-1, 1)),
        x0=psi0.reshape((-1, 1)),
        N=np.prod(LHS.N),
        **linear_solver_opts
    )[0].reshape(LHS.N)

    # Calculate error
    error = tn.linalg.norm((psi - psi0).flatten(), 2) / tn.linalg.norm(
        psi0.flatten(), 2
    )

    # Print progress
    print(
        "Angular Flux L2-Error = {}, Elapsed Time = {}".format(
            round(error.item(), 8),
            round(time.time() - start, 3),
        )
    )

    # Exit iteration if we've converged
    if error < tols[0]:
        print(
            "-- Converged: k = {}, Elapsed Time = {}".format(
                round(time.time() - start, 3)
            )
        )

    # Move operators back to CPU
    LHS = LHS.cpu()
    RHS = tuple([RHS[i].cpu() for i in range(len(RHS))])
    psi0 = psi0.cpu()
    psi = psi.cpu()

    return psi
