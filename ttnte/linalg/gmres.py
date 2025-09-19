from typing import Union, Optional, Callable, Literal
import time

import torch as tn
import numpy as np
from torchtt._iterative_solvers import gmres_restart

from ttnte.linalg import LinearOperator, Operator


def gmres(
    A: Union[Operator, LinearOperator],
    b: tn.Tensor,
    x0: Optional[tn.Tensor] = None,
    gpu_idx: Optional[int] = None,
    tol: float = 1e-5,
    atol: float = 0.0,
    restart: int = 20,
    maxiter: Optional[int] = None,
    solve_method: Literal["batched", "incremental"] = "batched",
    callback: Optional[Callable] = None,
    callback_frequency: int = 1,
    verbose: bool = True,
):
    """
    Generalized minimal residual method (GMRES) for solving linear systems.

    Parameters
    ----------
    A: ttnte.linalg.Operator or ttnte.linalg.LinearOperator
        Operator with an ``apply()`` method.
    b: torch.Tensor
        The left hand side of the equation.
    x0: torch.Tensor or None, default=None
        An initial guess for the solution.
    gpu_idx: int or None, default=None
        If ``gpu_idx`` is not None then GMRES is run on GPU with the given
        index.
    tol: float, default=1e-5
        Tolerance of GMRES with convergence condition
        ``norm(residual) <= max(tol*norm(b), atol)``.
    atol: float, default=0.0
        Tolerance of GMRES with convergence condition
        ``norm(residual) <= max(tol*norm(b), atol)``.
    restart: int, default=20
        Size of the Krylov subspace ("number of iterations") built between
        restarts.
    maxiter: int or None, default=None
        Maximum number of times to rebuild the size-``restart`` Krylov space
        starting from the solution found at the last iteration.
    solve_method: "batched" or "incremental"
        The "incremental" solve method builds a QR decomposition for the
        Krylov subspace incrementally using Givens rotation. This enables
        early stopping and is overall more stable than the "batched"
        method. The "batched" solution method solves a least squares problem
        from scratch at the end of each GMRES iteration. This has less overhead
        on GPUs. The "batched" method is only implemented in the C++ backend.
    callback: callable or None, default=None
        A callable called after each restart. This callable takes three inputs:
        iteration index ``i``, the residual norm for that iteration ``rnorm``,
        and the solution for that iteration ``x``.
    callback_frequency: int, default=1
        How often updates are printed to terminal and how often the callback
        is called.
    verbose: bool, default=True
        Whether to print the progress of GMRES.

    Returns
    -------
    x: torch.Tensor
        Solution.
    residual_norms: torch.Tensor
        Residual norms of maximum length ``maxiter + 1``. This is only returned in
        the C++ backend.
    """
    n = np.prod(b.shape)

    # Do some checks
    if np.prod(A.input_shape) != n or np.prod(A.output_shape) != n:
        raise RuntimeError("A must be square and match b")
    if solve_method != "batched" and solve_method != "incremental":
        raise RuntimeError("solve_method must be either 'batched' or 'incremental'")

    # Reshape and get initial guess
    b = b.reshape((-1, 1))
    x = x0.reshape((-1, 1)) if x0 is not None else tn.zeros_like(b, dtype=b.dtype)

    # Send operators and other information to GPU
    if gpu_idx is not None:
        A.cuda(gpu_idx)
        b = b.cuda(gpu_idx)
        x = x.cuda(gpu_idx)

    start = time.time()
    if verbose:
        print(
            f"Running {solve_method} GMRES on "
            + ("CPU" if gpu_idx is None else f"GPU {gpu_idx}")
        )

    # Ensure A outputs the right shape for torchtt
    A.matvec = lambda x: A.apply(x).reshape((-1, 1))

    # Run GMRES
    x = gmres_restart(
        LinOp=A,
        b=b,
        x0=x,
        N=n,
        max_iterations=restart,
        threshold=tol,
        resets=restart,
    )[0].reshape((-1, 1))

    if verbose:
        print(f"GMRES Finished\nElapsed Time: {time.time() - start}")

    if gpu_idx is not None:
        A.cpu()
        b = b.cpu()
        x = x.cpu()

    return x, None
