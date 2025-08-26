import time
from typing import Optional

import numpy as np
import torch as tn
from torchtt._iterative_solvers import gmres_restart

from ttnte.linalg.linear_operator import LinearOperator


def fixed_source(
    T: LinearOperator,
    q: tn.Tensor,
    tol: float = 1e-8,
    max_iters: int = 50,
    restarts: int = 100,
    device: Optional[int] = None,
):
    """
    Solve a fixed source calculation or :math:`Ax = b` problem.

    Parameters
    ----------
    TT: ttnte.linalg.LinearOperator
        Left hand side operator.
    q: torch.Tensor
        Source or right hand side of linear system.
    tol: float, default=1e-8
        Tolerance for GMRES.
    max_iters: int, default=50
        Max iterations for GMRES.
    restarts: int, default=100
        Number of restarts for GMRES.
    device: int, default=None
        Which device to run on.
    """
    # Send operators to GPU
    if device is not None:
        T = T.cuda(device)
        q = q.cuda(device)

    # Create initial guess
    psi0 = tn.ones(T.M, device=device).flatten()
    psi0 /= tn.linalg.norm(psi0, 2)
    psi = None

    start = time.time()
    # Run GMRES
    psi = gmres_restart(
        LinOp=T,
        b=q.reshape((-1, 1)),
        x0=psi0.reshape((-1, 1)),
        N=np.prod(T.N),
        max_iterations=max_iters,
        threshold=tol,
        resets=restarts,
    )[0].reshape(T.N)

    # Print finish
    print(f"GMRES Finished\nElapsed Time: {time.time() - start}")

    # Move operators back to CPU
    T = T.cpu()
    q = q.cpu()
    psi = psi.cpu()
    del psi0

    return psi
